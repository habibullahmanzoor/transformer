"""
DEBUG: Identify why UAAP-GEN v2.0 only achieves 48.5% fooling rate
Checks gradient flow, perturbation application, and optimization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms

# Set device
device = torch.device('cpu')
print(f"Using device: {device}")

# ============================================================================
# SIMPLE VIT FOR DEBUGGING
# ============================================================================

class DebugViT(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplest possible: 2-layer, 16 patches (7x7), 32 embed_dim, 4 heads
        self.patch_size = 7
        self.n_patches = (28 // 7) ** 2  # 16
        self.embed_dim = 32
        self.num_heads = 4
        
        self.patch_embed = nn.Conv2d(1, self.embed_dim, kernel_size=7, stride=7)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, self.embed_dim))
        
        # Single transformer block for debugging
        self.block = nn.ModuleDict({
            'norm1': nn.LayerNorm(self.embed_dim),
            'attn': nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True),
            'norm2': nn.LayerNorm(self.embed_dim),
            'mlp': nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 2),
                nn.GELU(),
                nn.Linear(self.embed_dim * 2, self.embed_dim)
            )
        })
        
        self.head = nn.Linear(self.embed_dim, 10)
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Debug: Store attention patterns
        self.last_attention = None
    
    def forward(self, x, perturbation=None):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, 32, 4, 4)
        x = x.flatten(2).transpose(1, 2)  # (B, 16, 32)
        
        # Add class token and position embedding
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, 17, 32)
        x = x + self.pos_embed
        
        # Transformer block
        residual = x
        x = self.block['norm1'](x)
        
        if perturbation is not None:
            # Manual attention with perturbation
            q = k = v = x
            scale = x.shape[-1] ** -0.5
            attn_scores = (q @ k.transpose(-2, -1)) * scale
            
            # Store clean attention
            self.last_attention = attn_scores.detach()
            
            # Apply perturbation
            attn_scores = attn_scores + perturbation
            
            # Softmax
            attn = attn_scores.softmax(dim=-1)
            x = (attn @ v)
        else:
            # Standard attention
            attn_output, attn_weights = self.block['attn'](x, x, x)
            self.last_attention = attn_weights.detach()
            x = attn_output
        
        x = residual + x
        
        # MLP
        residual = x
        x = self.block['norm2'](x)
        x = self.block['mlp'](x)
        x = residual + x
        
        # Classification
        x = x[:, 0]
        x = self.head(x)
        return x


# ============================================================================
# DEBUG TESTS
# ============================================================================

def test_gradient_flow():
    """Test if gradients flow through the model."""
    print("\n" + "="*70)
    print("TEST 1: Gradient Flow")
    print("="*70)
    
    model = DebugViT().to(device)
    
    # Test 1: Input gradients
    print("\n1. Testing input gradients...")
    x = torch.randn(2, 1, 28, 28, requires_grad=True, device=device)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    if x.grad is not None and x.grad.abs().sum() > 0:
        print("   ✅ Input gradients: OK")
    else:
        print("   ❌ Input gradients: FAILED")
        return False
    
    # Test 2: Parameter gradients
    print("\n2. Testing parameter gradients...")
    x = torch.randn(2, 1, 28, 28, device=device)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    params_with_grad = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            if p.grad.abs().sum() > 0:
                params_with_grad.append(name)
    
    if len(params_with_grad) > 0:
        print(f"   ✅ {len(params_with_grad)} parameters have gradients")
    else:
        print("   ❌ No parameters have gradients")
        return False
    
    return True


def test_perturbation_application():
    """Test if perturbation is actually applied."""
    print("\n" + "="*70)
    print("TEST 2: Perturbation Application")
    print("="*70)
    
    model = DebugViT().to(device)
    
    # Create a simple perturbation
    perturbation = torch.randn(17, 17, device=device) * 0.1
    
    # Test with and without perturbation
    x = torch.randn(1, 1, 28, 28, device=device)
    
    with torch.no_grad():
        out_clean = model(x)
        out_perturbed = model(x, perturbation)
    
    # Check if outputs are different
    diff = (out_clean - out_perturbed).abs().max().item()
    
    if diff > 1e-6:
        print(f"   ✅ Perturbation applied: output diff = {diff:.6f}")
        return True
    else:
        print(f"   ❌ Perturbation NOT applied: output diff = {diff:.6f}")
        return False


def test_perturbation_gradient():
    """Test if perturbation receives gradients."""
    print("\n" + "="*70)
    print("TEST 3: Perturbation Gradient")
    print("="*70)
    
    model = DebugViT().to(device)
    
    # Create perturbation parameter
    perturbation = torch.randn(17, 17, device=device, requires_grad=True) * 0.01
    
    x = torch.randn(1, 1, 28, 28, device=device)
    labels = torch.tensor([0], device=device)
    
    # Forward pass with perturbation
    out = model(x, perturbation)
    loss = F.cross_entropy(out, labels)
    
    # Backward
    loss.backward()
    
    if perturbation.grad is not None and perturbation.grad.abs().sum() > 0:
        print(f"   ✅ Perturbation has gradients: {perturbation.grad.abs().sum():.6f}")
        return True
    else:
        print("   ❌ Perturbation has NO gradients")
        return False


def test_uaap_optimization():
    """Test if UAAP optimization actually changes the perturbation."""
    print("\n" + "="*70)
    print("TEST 4: UAAP Optimization")
    print("="*70)
    
    model = DebugViT().to(device)
    
    # Create perturbation
    perturbation = torch.randn(17, 17, device=device, requires_grad=True) * 0.01
    initial_perturbation = perturbation.clone()
    
    optimizer = optim.Adam([perturbation], lr=0.01)
    
    # Load a small dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='/workspace/uaap/data', train=True, download=True, transform=transform)
    train_subset = torch.utils.data.Subset(train_dataset, list(range(10)))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=2, shuffle=True)
    
    # Optimization loop
    for iteration in range(5):
        optimizer.zero_grad()
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Clean forward
            model.eval()
            clean_out = model(images)
            clean_loss = F.cross_entropy(clean_out, labels)
            
            # Adversarial forward
            adv_out = model(images, perturbation)
            adv_loss = F.cross_entropy(adv_out, labels)
            
            # Fooling loss
            fooling_loss = adv_loss - clean_loss
            
            # Backward
            (-fooling_loss).backward()
        
        optimizer.step()
        
        # Check if perturbation changed
        diff = (perturbation - initial_perturbation).abs().sum().item()
        print(f"   Iter {iteration + 1}: Perturbation change = {diff:.6f}")
    
    final_diff = (perturbation - initial_perturbation).abs().sum().item()
    
    if final_diff > 1e-6:
        print(f"   ✅ Perturbation optimized: total change = {final_diff:.6f}")
        return True
    else:
        print(f"   ❌ Perturbation NOT optimized: total change = {final_diff:.6f}")
        return False


def test_fooling_effect():
    """Test if UAAP actually fools the model."""
    print("\n" + "="*70)
    print("TEST 5: Fooling Effect")
    print("="*70)
    
    model = DebugViT().to(device)
    
    # Load a small test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='/workspace/uaap/data', train=False, download=True, transform=transform)
    test_subset = torch.utils.data.Subset(test_dataset, list(range(10)))
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=10, shuffle=False)
    
    # Train model first
    print("\n  Training model...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):
        model.train()
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate clean
    model.eval()
    with torch.no_grad():
        clean_correct = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = torch.argmax(model(images), dim=1)
            clean_correct += (preds == labels).sum().item()
        clean_acc = clean_correct / len(test_subset)
    
    print(f"  Clean accuracy: {clean_acc:.4f}")
    
    # Generate UAAP
    print("\n  Generating UAAP...")
    perturbation = torch.randn(17, 17, device=device, requires_grad=True) * 0.01
    optimizer = optim.Adam([perturbation], lr=0.01)
    
    for iteration in range(10):
        optimizer.zero_grad()
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            clean_out = model(images)
            clean_loss = F.cross_entropy(clean_out, labels)
            adv_out = model(images, perturbation)
            adv_loss = F.cross_entropy(adv_out, labels)
            (- (adv_loss - clean_loss)).backward()
        optimizer.step()
        with torch.no_grad():
            for p in [perturbation]:
                norm = torch.norm(p)
                if norm > 0.5:
                    p.data *= (0.5 / (norm + 1e-8))
    
    # Evaluate
    with torch.no_grad():
        adv_correct = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = torch.argmax(model(images, perturbation), dim=1)
            adv_correct += (preds == labels).sum().item()
        adv_acc = adv_correct / len(test_subset)
    
    fooling_rate = clean_acc - adv_acc
    
    print(f"  Adversarial accuracy: {adv_acc:.4f}")
    print(f"  Fooling rate: {fooling_rate:.4f}")
    
    if fooling_rate > 0.5:
        print(f"   ✅ Fooling rate > 50%")
        return True
    else:
        print(f"   ❌ Fooling rate too low")
        return False


def main():
    print("\n" + "="*70)
    print("UAAP-GEN DEBUG: Identifying Issues")
    print("="*70)
    
    results = []
    
    # Run all tests
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("Perturbation Application", test_perturbation_application()))
    results.append(("Perturbation Gradient", test_perturbation_gradient()))
    results.append(("UAAP Optimization", test_uaap_optimization()))
    results.append(("Fooling Effect", test_fooling_effect()))
    
    # Summary
    print("\n" + "="*70)
    print("DEBUG SUMMARY")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    # Diagnosis
    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print("="*70)
    
    if failed == 0:
        print("\n✅ All tests passed! Issue may be elsewhere.")
        print("   Check: model capacity, training epochs, dataset size")
    else:
        print("\n❌ Some tests failed. Issues found:")
        for test_name, result in results:
            if not result:
                print(f"   - {test_name}")
        
        print("\nRecommended fixes:")
        if not results[0][1]:  # Gradient flow failed
            print("   1. Check model architecture - gradients may not flow")
        if not results[1][1]:  # Perturbation not applied
            print("   2. Check perturbation application in forward pass")
        if not results[2][1]:  # Perturbation no gradient
            print("   3. Check if perturbation is in computation graph")
        if not results[3][1]:  # Optimization failed
            print("   4. Check optimization loop - gradients may not accumulate")
        if not results[4][1]:  # Fooling too low
            print("   5. If all above pass, issue is in model capacity or training")


if __name__ == '__main__':
    main()
