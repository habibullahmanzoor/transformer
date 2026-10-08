"""
Single test to validate the fixed UAAP-GEN v2.0 implementation
Tests on MNIST with one seed to verify Phase 1 fixes work
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import time

# Set device
device = torch.device('cpu')
print(f"Using device: {device}")

# ============================================================================
# VIT MODEL
# ============================================================================

class ViT(nn.Module):
    def __init__(self, img_size=28, patch_size=7, num_classes=10, 
                 embed_dim=64, num_heads=4, num_layers=2, in_channels=1):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.GELU(),
                    nn.Linear(embed_dim * 2, embed_dim)
                )
            })
            for _ in range(num_layers)
        ])
        
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x, attn_perturbations=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        
        for idx, block in enumerate(self.blocks):
            residual = x
            x = block['norm1'](x)
            if attn_perturbations and idx < len(attn_perturbations):
                perturbation = attn_perturbations[idx]
                q = k = v = x
                scale = x.shape[-1] ** -0.5
                attn_scores = (q @ k.transpose(-2, -1)) * scale
                attn_scores = attn_scores + perturbation
                attn = attn_scores.softmax(dim=-1)
                x = (attn @ v)
            else:
                attn_output, _ = block['attn'](x, x, x)
                x = attn_output
            x = residual + x
            residual = x
            x = block['norm2'](x)
            x = block['mlp'](x)
            x = residual + x
        
        x = x[:, 0]
        x = self.head(x)
        return x


# ============================================================================
# UAAP GENERATOR - FIXED
# ============================================================================

class UAAPGenerator:
    def __init__(self, model, epsilon=0.5, num_iter=20, lr=0.01, device='cpu'):
        self.model = model.to(device)
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.lr = lr
        self.device = device
        self.num_layers = len(model.blocks)
        self.n_tokens = model.n_patches + 1
        
        self.perturbations = nn.ParameterList([
            nn.Parameter(torch.randn(self.n_tokens, self.n_tokens, device=device) * 0.01)
            for _ in range(self.num_layers)
        ])
    
    def compute_fooling_loss(self, images, labels):
        self.model.eval()
        clean_outputs = self.model(images)
        clean_loss = F.cross_entropy(clean_outputs, labels)
        layer_perturbations = [p for p in self.perturbations]
        adv_outputs = self.model(images, layer_perturbations)
        adv_loss = F.cross_entropy(adv_outputs, labels)
        return adv_loss - clean_loss
    
    def generate(self, dataloader):
        """FIXED: Dataset-wide optimization"""
        optimizer = optim.Adam(self.perturbations, lr=self.lr)
        
        for iteration in range(self.num_iter):
            # Accumulate gradients across entire dataset
            optimizer.zero_grad()
            
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                batch_fooling = self.compute_fooling_loss(images, labels)
                # Maximize fooling = minimize negative fooling
                (-batch_fooling).backward()
            
            # Single optimization step for entire dataset
            optimizer.step()
            
            # Strict PGD projection
            with torch.no_grad():
                for p in self.perturbations:
                    norm = torch.norm(p)
                    if norm > self.epsilon:
                        p.data *= (self.epsilon / (norm + 1e-8))
            
            if (iteration + 1) % 5 == 0:
                # Compute average fooling for display
                total_fooling = 0
                num_batches = 0
                for images, labels in dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    total_fooling += self.compute_fooling_loss(images, labels).item()
                    num_batches += 1
                avg_fooling = total_fooling / num_batches if num_batches > 0 else 0
                print(f"    Iter {iteration + 1}/{self.num_iter}, Avg Fooling: {avg_fooling:.4f}")
        
        return [p.detach() for p in self.perturbations]
    
    def evaluate(self, dataloader):
        self.model.eval()
        layer_perturbations = [p.detach() for p in self.perturbations]
        clean_correct = adv_correct = total = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                clean_preds = torch.argmax(self.model(images), dim=1)
                adv_preds = torch.argmax(self.model(images, layer_perturbations), dim=1)
                clean_correct += (clean_preds == labels).sum().item()
                adv_correct += (adv_preds == labels).sum().item()
                total += len(labels)
        clean_acc = clean_correct / total if total > 0 else 0
        adv_acc = adv_correct / total if total > 0 else 0
        fooling_rate = clean_acc - adv_acc
        return clean_acc, adv_acc, fooling_rate


# ============================================================================
# MAIN TEST
# ============================================================================

def main():
    print("\n" + "="*70)
    print("UAAP-GEN v2.0: Single Test Validation")
    print("Testing Phase 1 fixes on MNIST")
    print("="*70)
    
    # Load MNIST
    print("\nLoading MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='/workspace/uaap/data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='/workspace/uaap/data', train=False, download=True, transform=transform)
    
    # Use subset for fast testing
    train_subset = torch.utils.data.Subset(train_dataset, list(range(500)))
    test_subset = torch.utils.data.Subset(test_dataset, list(range(200)))
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=32, shuffle=False)
    
    print(f"  Train: {len(train_subset)}, Test: {len(test_subset)}")
    
    # Create and train model
    print("\nCreating model...")
    model = ViT(img_size=28, patch_size=7, num_classes=10, 
               embed_dim=64, num_heads=4, num_layers=2, in_channels=1)
    model.to(device)
    
    print("Training...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                preds = torch.argmax(model(images), dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        acc = correct / total if total > 0 else 0
        print(f"  Epoch {epoch + 1}/10, Test Acc: {acc:.4f}")
    
    # Evaluate clean accuracy
    clean_acc = acc
    print(f"\nFinal clean accuracy: {clean_acc:.4f}")
    
    # Generate UAAP
    print("\nGenerating UAAP...")
    uaap_gen = UAAPGenerator(model=model, epsilon=0.5, num_iter=20, lr=0.01, device=device)
    perturbations = uaap_gen.generate(train_loader)
    
    # Evaluate
    print("\nEvaluating UAAP...")
    clean_acc_final, adv_acc, fooling_rate = uaap_gen.evaluate(test_loader)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    print(f"Clean Accuracy:     {clean_acc_final:.4f}")
    print(f"Adversarial Acc:    {adv_acc:.4f}")
    print(f"Fooling Rate:       {fooling_rate:.4f}")
    
    # Assessment
    print(f"\n{'='*70}")
    print("ASSESSMENT")
    print("="*70)
    
    if fooling_rate > 0.95:
        print("\n✅ EXCELLENT: >95% fooling rate!")
        print("   Phase 1 fixes working perfectly.")
        print("   ✅ Ready for Phase 2")
    elif fooling_rate > 0.7:
        print(f"\n✅ GOOD: {fooling_rate:.1%} fooling rate")
        print("   Phase 1 fixes partially working.")
        print("   ⚠️  May need investigation")
    elif fooling_rate > 0.5:
        print(f"\n⚠️  ACCEPTABLE: {fooling_rate:.1%} fooling rate")
        print("   But below expected >95%")
        print("   ❌ Need to debug")
    else:
        print(f"\n❌ POOR: Only {fooling_rate:.1%} fooling rate")
        print("   Phase 1 fixes not working")
        print("   ❌ Critical debugging needed")
    
    return fooling_rate


if __name__ == '__main__':
    start_time = time.time()
    fooling_rate = main()
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
