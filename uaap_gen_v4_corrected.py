"""
UAAP-GEN v4.0: CORRECTED IMPLEMENTATION
Fixes all critical issues identified in the research assessment.

PROBLEM 1 FIXED: Perturbations applied INSIDE original MultiheadAttention
- Preserves all Q,K,V projections and output projection
- f(x, delta=0) == f(x, clean) is guaranteed

PROBLEM 2 FIXED: Input-UAP preprocessing corrected
- Perturb in raw pixel space [0, 1]
- Normalize AFTER perturbation
- Consistent L_infinity budget

PROBLEM 3 FIXED: Correct metrics implementation
- FR (Fooling Rate): prediction-change rate
- ASR (Attack Success Rate): among correctly classified samples
- Accuracy reduction: clean_acc - adv_acc

PROBLEM 4 FIXED: Consistent threat models
- UAAP-GEN: L2 budget on attention logits
- Input-UAP: L_infinity budget on pixels

Author: Vibe Code (Mistral AI)
Date: 2025
Publication Target: ICLR/NeurIPS/CVPR 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import os
import json
from datetime import datetime

# Set device
device = torch.device('cpu')
print(f"Using device: {device}")

# Create directories
os.makedirs('/workspace/uaap/results/v4_corrected', exist_ok=True)


# ============================================================================
# CUSTOM MULTIHEAD ATTENTION WITH PERTURBATION SUPPORT
# ============================================================================

class PerturbableMultiheadAttention(nn.Module):
    """
    Custom MultiheadAttention that supports adding perturbations to attention scores.
    
    CRITICAL: This preserves ALL original computations when perturbation=0.
    - Q, K, V projections
    - Output projection  
    - Multi-head structure
    - When perturbation is None or zero, output is IDENTICAL to standard MultiheadAttention
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        
        # Projections (same as PyTorch's MultiheadAttention)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights to match PyTorch's default
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(self, x, perturbation=None):
        """
        Forward pass with optional perturbation.
        
        Args:
            x: Input tensor (B, N, E)
            perturbation: Optional attention score perturbation (num_heads, N, N)
                         If provided, added to attention scores before softmax
        
        Returns:
            Output tensor with same shape as x
        
        GUARANTEE: When perturbation is None or zero, output equals standard MHA
        """
        B, N, E = x.shape
        
        # Project Q, K, V (same as PyTorch)
        q = self.q_proj(x)  # (B, N, E)
        k = self.k_proj(x)  # (B, N, E)
        v = self.v_proj(x)  # (B, N, E)
        
        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        
        # Compute attention scores
        scale = (self.head_dim) ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, num_heads, N, N)
        
        # CRITICAL: Add perturbation INSIDE the attention computation
        # This preserves all projections and structure
        if perturbation is not None:
            # perturbation shape: (num_heads, N, N)
            # Broadcast to (B, num_heads, N, N)
            attn_scores = attn_scores + perturbation.unsqueeze(0)
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # (B, num_heads, N, head_dim)
        
        # Reshape back and apply output projection
        output = output.transpose(1, 2).contiguous().view(B, N, E)
        output = self.out_proj(output)
        
        return output


# ============================================================================
# 1. VISION TRANSFORMER - CORRECTED
# ============================================================================

class ViT(nn.Module):
    """
    Vision Transformer with perturbable attention.
    
    CRITICAL: Uses PerturbableMultiheadAttention which preserves
    all original computations when perturbation=0.
    """
    def __init__(self, img_size=28, patch_size=7, num_classes=10, 
                 embed_dim=64, num_heads=4, num_layers=2, in_channels=1):
        super().__init__()
        
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.num_heads = num_heads
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        # Transformer blocks with perturbable attention
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': PerturbableMultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.GELU(),
                    nn.Linear(embed_dim * 2, embed_dim)
                )
            })
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize all linear layers
        for block in self.blocks:
            nn.init.xavier_uniform_(block['attn'].q_proj.weight)
            nn.init.xavier_uniform_(block['attn'].k_proj.weight)
            nn.init.xavier_uniform_(block['attn'].v_proj.weight)
            nn.init.xavier_uniform_(block['attn'].out_proj.weight)
    
    def forward(self, x, attn_perturbations=None):
        """
        Forward pass with optional attention perturbations.
        
        CRITICAL GUARANTEE: When attn_perturbations is None or all zeros,
        this produces EXACTLY the same output as without perturbations.
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, C, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        
        # Add class token and position embedding
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, n_patches+1, embed_dim)
        x = x + self.pos_embed
        
        # Transformer blocks
        for idx, block in enumerate(self.blocks):
            residual = x
            x = block['norm1'](x)
            
            # Self-attention with optional perturbation
            if attn_perturbations is not None and idx < len(attn_perturbations):
                perturbation = attn_perturbations[idx]
            else:
                perturbation = None
            
            x = block['attn'](x, perturbation)
            x = residual + x
            
            # MLP
            residual = x
            x = block['norm2'](x)
            x = block['mlp'](x)
            x = residual + x
        
        # Classification
        x = x[:, 0]  # Take class token
        x = self.head(x)
        return x


# ============================================================================
# 2. UAAP GENERATOR - CORRECTED
# ============================================================================

class UAAPGenerator:
    """
    Universal Adversarial Attention Perturbations Generator - CORRECTED
    
    Key improvements:
    1. Perturbations applied INSIDE original attention computation
    2. Dataset-wide optimization with proper gradient accumulation
    3. Strict PGD projection (L2 norm)
    4. Correct metrics: FR, ASR, accuracy reduction
    """
    
    def __init__(self, model, epsilon=0.5, num_iter=20, lr=0.1, device='cpu'):
        self.model = model.to(device)
        self.epsilon = epsilon  # L2 norm budget for attention perturbations
        self.num_iter = num_iter
        self.lr = lr
        self.device = device
        
        # Get model dimensions
        self.num_layers = len(model.blocks)
        self.num_heads = model.num_heads
        self.n_tokens = model.n_patches + 1
        
        # Create perturbations as LEAF tensors
        # Shape: (num_layers, num_heads, n_tokens, n_tokens)
        self.perturbations = nn.ParameterList([
            nn.Parameter(torch.randn(self.num_heads, self.n_tokens, self.n_tokens, device=device) * 0.01)
            for _ in range(self.num_layers)
        ])
        
        print(f"  UAAP Generator: {self.num_layers} layers, {self.num_heads} heads, {self.n_tokens} tokens")
        print(f"  Perturbation shape per layer: ({self.num_heads}, {self.n_tokens}, {self.n_tokens})")
        print(f"  Total perturbation params: {sum(p.numel() for p in self.perturbations):,}")
    
    def compute_batch_loss(self, images, labels, normalize_fn=None):
        """Compute attack loss for a single batch."""
        self.model.eval()
        
        # Apply normalization if provided
        if normalize_fn is not None:
            images = normalize_fn(images)
        
        # Clean forward pass
        clean_outputs = self.model(images)
        clean_loss = F.cross_entropy(clean_outputs, labels)
        
        # Adversarial forward pass with perturbations
        layer_perturbations = [p for p in self.perturbations]
        adv_outputs = self.model(images, layer_perturbations)
        adv_loss = F.cross_entropy(adv_outputs, labels)
        
        # Attack objective: maximize (adv_loss - clean_loss)
        attack_loss = adv_loss - clean_loss
        
        return attack_loss
    
    def generate(self, dataloader, normalize_fn=None, verbose=True):
        """
        Generate UAAP by optimizing across the ENTIRE dataset.
        
        Uses dataset-wide gradient accumulation for true universal perturbation.
        """
        optimizer = optim.Adam(self.perturbations, lr=self.lr, weight_decay=0)
        
        for iteration in range(self.num_iter):
            optimizer.zero_grad()
            
            # Accumulate gradients across entire dataset
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                batch_loss = self.compute_batch_loss(images, labels, normalize_fn)
                (-batch_loss).backward()  # Maximize attack loss
            
            # Single optimization step
            optimizer.step()
            
            # Strict PGD projection (L2 norm per layer)
            with torch.no_grad():
                for p in self.perturbations:
                    norm = torch.norm(p)
                    if norm > self.epsilon:
                        p.data *= (self.epsilon / (norm + 1e-8))
            
            if verbose and (iteration + 1) % 5 == 0:
                # Compute metrics for display
                clean_acc, adv_acc, fooling_rate, asr, fr = self.evaluate(dataloader, normalize_fn)
                print(f"    Iter {iteration + 1}/{self.num_iter}, "
                      f"Clean={clean_acc:.4f}, Adv={adv_acc:.4f}, "
                      f"FR={fr:.4f}, ASR={asr:.4f}")
        
        return [p.detach() for p in self.perturbations]
    
    def evaluate(self, dataloader, normalize_fn=None):
        """
        Evaluate UAAP performance with CORRECTED metrics.
        
        Returns:
            clean_acc: Clean accuracy
            adv_acc: Adversarial accuracy  
            fooling_rate: Accuracy reduction (clean_acc - adv_acc)
            asr: Attack Success Rate (among correctly classified)
            fr: Fooling Rate (prediction-change rate)
        """
        self.model.eval()
        layer_perturbations = [p.detach() for p in self.perturbations]
        
        clean_correct = adv_correct = total = 0
        prediction_changes = 0
        correctly_classified = 0
        asr_numerator = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Apply normalization if provided
                if normalize_fn is not None:
                    images = normalize_fn(images)
                
                clean_preds = torch.argmax(self.model(images), dim=1)
                adv_preds = torch.argmax(self.model(images, layer_perturbations), dim=1)
                
                clean_correct += (clean_preds == labels).sum().item()
                adv_correct += (adv_preds == labels).sum().item()
                total += len(labels)
                
                # Prediction-change rate (FR)
                prediction_changes += (clean_preds != adv_preds).sum().item()
                
                # ASR: among correctly classified samples
                correctly_classified += (clean_preds == labels).sum().item()
                asr_numerator += ((clean_preds == labels) & (adv_preds != labels)).sum().item()
        
        clean_acc = clean_correct / total if total > 0 else 0
        adv_acc = adv_correct / total if total > 0 else 0
        fooling_rate = clean_acc - adv_acc  # Accuracy reduction
        fr = prediction_changes / total if total > 0 else 0  # Prediction-change rate
        asr = asr_numerator / correctly_classified if correctly_classified > 0 else 0  # Attack Success Rate
        
        return clean_acc, adv_acc, fooling_rate, asr, fr


# ============================================================================
# 3. INPUT-UAP BASELINE - CORRECTED
# ============================================================================

class InputUAPGenerator:
    """
    Standard Input-Level Universal Adversarial Perturbation - CORRECTED.
    
    Key fixes:
    1. Perturb in RAW pixel space [0, 1]
    2. Normalize AFTER perturbation
    3. Consistent L_infinity budget
    """
    
    def __init__(self, model, epsilon=0.031, num_iter=20, lr=0.1, device='cpu'):
        self.model = model.to(device)
        self.epsilon = epsilon  # L_infinity budget (8/255 ≈ 0.031 for MNIST)
        self.num_iter = num_iter
        self.lr = lr
        self.device = device
        self.perturbation = None
    
    def generate(self, dataloader, img_shape, normalize_fn, verbose=True):
        """
        Generate input-level UAP.
        
        CORRECTED: 
        1. Perturb in raw pixel space
        2. Apply normalization AFTER perturbation
        """
        in_channels, img_h, img_w = img_shape
        
        # Initialize perturbation as LEAF tensor (raw pixel space)
        self.perturbation = nn.Parameter(
            torch.randn(1, in_channels, img_h, img_w, device=self.device) * 0.01
        )
        
        optimizer = optim.Adam([self.perturbation], lr=self.lr, weight_decay=0)
        
        for iteration in range(self.num_iter):
            optimizer.zero_grad()
            
            for images_raw, labels in dataloader:
                images_raw = images_raw.to(self.device)
                labels = labels.to(self.device)
                
                # Clean forward (with normalization)
                self.model.eval()
                images_clean_norm = normalize_fn(images_raw)
                clean_outputs = self.model(images_clean_norm)
                clean_loss = F.cross_entropy(clean_outputs, labels)
                
                # Adversarial forward
                # 1. Add perturbation in RAW space
                adv_images_raw = images_raw + self.perturbation
                # 2. Clamp to valid pixel range [0, 1]
                adv_images_raw = torch.clamp(adv_images_raw, 0, 1)
                # 3. Normalize AFTER perturbation
                adv_images_norm = normalize_fn(adv_images_raw)
                adv_outputs = self.model(adv_images_norm)
                adv_loss = F.cross_entropy(adv_outputs, labels)
                
                # Attack objective
                attack_loss = adv_loss - clean_loss
                
                (-attack_loss).backward()
            
            optimizer.step()
            
            # Project to L_infinity epsilon ball
            with torch.no_grad():
                self.perturbation.data = torch.clamp(
                    self.perturbation.data, 
                    -self.epsilon, 
                    self.epsilon
                )
            
            if verbose and (iteration + 1) % 5 == 0:
                clean_acc, adv_acc, fooling_rate, asr, fr = self.evaluate(dataloader, normalize_fn)
                print(f"    Iter {iteration + 1}/{self.num_iter}, "
                      f"Clean={clean_acc:.4f}, Adv={adv_acc:.4f}, "
                      f"FR={fr:.4f}, ASR={asr:.4f}")
        
        return self.perturbation.detach()
    
    def evaluate(self, dataloader, normalize_fn):
        """Evaluate input-UAP performance with CORRECTED metrics."""
        self.model.eval()
        
        clean_correct = adv_correct = total = 0
        prediction_changes = 0
        correctly_classified = 0
        asr_numerator = 0
        
        with torch.no_grad():
            for images_raw, labels in dataloader:
                images_raw = images_raw.to(self.device)
                labels = labels.to(self.device)
                
                # Clean
                images_clean_norm = normalize_fn(images_raw)
                clean_preds = torch.argmax(self.model(images_clean_norm), dim=1)
                
                # Adversarial
                adv_images_raw = images_raw + self.perturbation
                adv_images_raw = torch.clamp(adv_images_raw, 0, 1)
                adv_images_norm = normalize_fn(adv_images_raw)
                adv_preds = torch.argmax(self.model(adv_images_norm), dim=1)
                
                clean_correct += (clean_preds == labels).sum().item()
                adv_correct += (adv_preds == labels).sum().item()
                total += len(labels)
                
                prediction_changes += (clean_preds != adv_preds).sum().item()
                correctly_classified += (clean_preds == labels).sum().item()
                asr_numerator += ((clean_preds == labels) & (adv_preds != labels)).sum().item()
        
        clean_acc = clean_correct / total if total > 0 else 0
        adv_acc = adv_correct / total if total > 0 else 0
        fooling_rate = clean_acc - adv_acc
        fr = prediction_changes / total if total > 0 else 0
        asr = asr_numerator / correctly_classified if correctly_classified > 0 else 0
        
        return clean_acc, adv_acc, fooling_rate, asr, fr


# ============================================================================
# 4. DATASET LOADERS - CORRECTED
# ============================================================================

def get_mnist_loaders(batch_size=32, subset_size=500):
    """
    Get MNIST loaders with raw images.
    
    Returns data in raw format [0, 1] for proper Input-UAP preprocessing.
    """
    # Raw transform (for Input-UAP) - returns tensor in [0, 1]
    raw_transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL to tensor [0, 1]
    ])
    
    # Normalized transform (for ViT) - applies to tensor
    norm_transform = transforms.Normalize((0.1307,), (0.3081,))
    
    train_dataset_raw = datasets.MNIST(root='/workspace/uaap/data', train=True, download=True, transform=raw_transform)
    test_dataset_raw = datasets.MNIST(root='/workspace/uaap/data', train=False, download=True, transform=raw_transform)
    
    train_subset = torch.utils.data.Subset(train_dataset_raw, list(range(min(subset_size, len(train_dataset_raw)))))
    test_subset = torch.utils.data.Subset(test_dataset_raw, list(range(min(200, len(test_dataset_raw)))))
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, (1, 28, 28), norm_transform


def get_fashion_mnist_loaders(batch_size=32, subset_size=500):
    """Get FashionMNIST loaders."""
    raw_transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL to tensor [0, 1]
    ])
    
    norm_transform = transforms.Normalize((0.1307,), (0.3081,))
    
    train_dataset_raw = datasets.FashionMNIST(root='/workspace/uaap/data', train=True, download=True, transform=raw_transform)
    test_dataset_raw = datasets.FashionMNIST(root='/workspace/uaap/data', train=False, download=True, transform=raw_transform)
    
    train_subset = torch.utils.data.Subset(train_dataset_raw, list(range(min(subset_size, len(train_dataset_raw)))))
    test_subset = torch.utils.data.Subset(test_dataset_raw, list(range(min(200, len(test_dataset_raw)))))
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, (1, 28, 28), norm_transform


# ============================================================================
# 5. VALIDATION TESTS
# ============================================================================

def test_zero_perturbation_equivalence(model, dataloader, norm_transform, device='cpu'):
    """
    CRITICAL VALIDATION TEST: Verify that f(x, delta=0) == f(x, clean)
    
    This ensures the perturbation implementation doesn't change the forward pass
    when perturbation is zero.
    """
    model.eval()
    model.to(device)
    
    print("\n" + "="*80)
    print("VALIDATION TEST: Zero Perturbation Equivalence")
    print("="*80)
    
    all_match = True
    max_diff = 0.0
    
    # Get model dimensions
    num_layers = len(model.blocks)
    num_heads = model.num_heads
    n_tokens = model.n_patches + 1
    
    with torch.no_grad():
        for images_raw, labels in dataloader:
            images_raw = images_raw.to(device)
            images_norm = norm_transform(images_raw)
            
            # Clean forward (no perturbations)
            clean_outputs = model(images_norm)
            
            # Forward with zero perturbations
            zero_perturbations = [
                torch.zeros(num_heads, n_tokens, n_tokens, device=device)
                for _ in range(num_layers)
            ]
            
            zero_outputs = model(images_norm, zero_perturbations)
            
            # Check if outputs match
            diff = torch.abs(clean_outputs - zero_outputs).max().item()
            max_diff = max(max_diff, diff)
            
            if diff > 1e-6:
                all_match = False
                print(f"  ❌ MISMATCH: max diff = {diff}")
            else:
                print(f"  ✅ MATCH: max diff = {diff:.2e}")
    
    if all_match:
        print(f"\n✅ PASSED: All outputs match (max diff: {max_diff:.2e})")
    else:
        print(f"\n❌ FAILED: Outputs differ (max diff: {max_diff})")
    
    return all_match, max_diff


def test_gradient_flow(model, dataloader, norm_transform, device='cpu'):
    """
    Test that gradients flow properly through perturbations.
    """
    print("\n" + "="*80)
    print("VALIDATION TEST: Gradient Flow")
    print("="*80)
    
    model.eval()
    model.to(device)
    
    # Create UAAP generator
    uaap_gen = UAAPGenerator(model, device=device)
    
    # Check if perturbations have gradients
    for i, p in enumerate(uaap_gen.perturbations):
        print(f"  Layer {i}: requires_grad={p.requires_grad}")
    
    # Do a forward and backward pass
    for images_raw, labels in dataloader:
        images_raw = images_raw.to(device)
        labels = labels.to(device)
        images_norm = norm_transform(images_raw)
        
        # Clean forward
        clean_outputs = model(images_norm)
        clean_loss = F.cross_entropy(clean_outputs, labels)
        
        # Adversarial forward
        layer_perturbations = [p for p in uaap_gen.perturbations]
        adv_outputs = model(images_norm, layer_perturbations)
        adv_loss = F.cross_entropy(adv_outputs, labels)
        
        # Attack loss
        attack_loss = adv_loss - clean_loss
        
        # Backward
        attack_loss.backward()
        
        # Check gradients
        has_grad = False
        for i, p in enumerate(uaap_gen.perturbations):
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                print(f"  Layer {i}: grad_norm = {grad_norm:.4f}")
                has_grad = True
            else:
                print(f"  Layer {i}: NO GRADIENT")
        
        break
    
    if has_grad:
        print("\n✅ PASSED: Gradients flow through perturbations")
    else:
        print("\n❌ FAILED: No gradients")
    
    return has_grad


# ============================================================================
# 6. MAIN EXPERIMENT
# ============================================================================

def run_experiment(dataset_name='MNIST', seed=42, num_layers=2, num_iter=20):
    """
    Run complete experiment with corrected implementation.
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: {dataset_name} | Seed={seed} | Layers={num_layers}")
    print("="*80)
    
    # Get dataset
    if dataset_name == 'MNIST':
        train_loader, test_loader, img_shape, norm_transform = get_mnist_loaders()
    elif dataset_name == 'FashionMNIST':
        train_loader, test_loader, img_shape, norm_transform = get_fashion_mnist_loaders()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"  Dataset: {dataset_name}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Image shape: {img_shape}")
    
    # Create model
    print("\n  Creating model...")
    model = ViT(img_size=img_shape[1], patch_size=7, num_classes=10,
                embed_dim=64, num_heads=4, num_layers=num_layers, in_channels=img_shape[0])
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    
    # Train model
    print("\n  Training model...")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    for epoch in range(30):
        for images_raw, labels in train_loader:
            images_raw = images_raw.to(device)
            labels = labels.to(device)
            images_norm = norm_transform(images_raw)
            
            optimizer.zero_grad()
            outputs = model(images_norm)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images_raw, labels in train_loader:
                    images_raw = images_raw.to(device)
                    labels = labels.to(device)
                    images_norm = norm_transform(images_raw)
                    outputs = model(images_norm)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += len(labels)
            acc = correct / total if total > 0 else 0
            print(f"    Epoch {epoch + 1}/30, Train Acc: {acc:.4f}")
            model.train()
    
    # Evaluate clean accuracy on test set
    print("\n  Evaluating clean accuracy on test set...")
    model.eval()
    clean_correct = 0
    total = 0
    with torch.no_grad():
        for images_raw, labels in test_loader:
            images_raw = images_raw.to(device)
            labels = labels.to(device)
            images_norm = norm_transform(images_raw)
            outputs = model(images_norm)
            preds = torch.argmax(outputs, dim=1)
            clean_correct += (preds == labels).sum().item()
            total += len(labels)
    clean_acc = clean_correct / total if total > 0 else 0
    print(f"  Clean Test Accuracy: {clean_acc:.4f}")
    
    # Run validation tests
    print("\n  Running validation tests...")
    test_zero_perturbation_equivalence(model, test_loader, norm_transform, device)
    test_gradient_flow(model, test_loader, norm_transform, device)
    
    # Generate and evaluate UAAP-GEN
    print("\n  Generating UAAP-GEN on TRAIN set...")
    uaap_gen = UAAPGenerator(model, epsilon=5.0, num_iter=num_iter, lr=0.1, device=device)
    uaap_perturbations = uaap_gen.generate(train_loader, norm_transform, verbose=True)
    print("\n  Evaluating UAAP-GEN on TEST set...")
    clean_acc_uaap, adv_acc_uaap, fooling_rate_uaap, asr_uaap, fr_uaap = uaap_gen.evaluate(test_loader, norm_transform)
    
    print(f"\n  UAAP-GEN Results:")
    print(f"    Clean Accuracy: {clean_acc_uaap:.4f}")
    print(f"    Adversarial Accuracy: {adv_acc_uaap:.4f}")
    print(f"    Accuracy Reduction: {fooling_rate_uaap:.4f}")
    print(f"    Fooling Rate (FR): {fr_uaap:.4f}")
    print(f"    Attack Success Rate (ASR): {asr_uaap:.4f}")
    
    # Generate and evaluate Input-UAP
    print("\n  Generating Input-UAP on TRAIN set...")
    input_uap_gen = InputUAPGenerator(model, epsilon=0.031, num_iter=num_iter, lr=0.1, device=device)
    input_uap = input_uap_gen.generate(train_loader, img_shape, norm_transform, verbose=True)
    print("\n  Evaluating Input-UAP on TEST set...")
    clean_acc_input, adv_acc_input, fooling_rate_input, asr_input, fr_input = input_uap_gen.evaluate(test_loader, norm_transform)
    
    print(f"\n  Input-UAP Results:")
    print(f"    Clean Accuracy: {clean_acc_input:.4f}")
    print(f"    Adversarial Accuracy: {adv_acc_input:.4f}")
    print(f"    Accuracy Reduction: {fooling_rate_input:.4f}")
    print(f"    Fooling Rate (FR): {fr_input:.4f}")
    print(f"    Attack Success Rate (ASR): {asr_input:.4f}")
    
    # Save results
    results = {
        'dataset': dataset_name,
        'seed': seed,
        'num_layers': num_layers,
        'num_iter': num_iter,
        'model_params': total_params,
        'clean_train_acc': acc,
        'clean_test_acc': clean_acc,
        'uaap_gen': {
            'clean_acc': clean_acc_uaap,
            'adv_acc': adv_acc_uaap,
            'fooling_rate': fooling_rate_uaap,
            'fr': fr_uaap,
            'asr': asr_uaap
        },
        'input_uap': {
            'clean_acc': clean_acc_input,
            'adv_acc': adv_acc_input,
            'fooling_rate': fooling_rate_input,
            'fr': fr_input,
            'asr': asr_input
        }
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f'/workspace/uaap/results/v4_corrected/results_{timestamp}_seed{seed}.json'
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {results_path}")
    
    return results


def main():
    """Run complete experiments for multiple seeds and datasets."""
    print("\n" + "="*80)
    print("UAAP-GEN v4.0: CORRECTED IMPLEMENTATION")
    print("Fixes all critical issues from research assessment")
    print("="*80)
    
    # Run experiments for multiple seeds
    seeds = [42, 123, 456]
    datasets = ['MNIST']  # Start with MNIST for validation
    
    all_results = []
    
    for dataset in datasets:
        for seed in seeds:
            try:
                results = run_experiment(dataset_name=dataset, seed=seed, num_layers=2, num_iter=20)
                all_results.append(results)
            except Exception as e:
                print(f"\n  ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # Save summary
    summary = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'total_experiments': len(all_results),
        'results': all_results
    }
    
    summary_path = f'/workspace/uaap/results/v4_corrected/summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n" + "="*80)
    print(f"Summary saved to: {summary_path}")
    print("="*80)
    
    # Print overall summary
    if all_results:
        print("\nOVERALL RESULTS:")
        print("-" * 80)
        for i, r in enumerate(all_results):
            print(f"\nExperiment {i+1}: {r['dataset']}, Seed={r['seed']}")
            print(f"  Clean Test Acc: {r['clean_test_acc']:.4f}")
            print(f"  UAAP-GEN: FR={r['uaap_gen']['fr']:.4f}, ASR={r['uaap_gen']['asr']:.4f}")
            print(f"  Input-UAP: FR={r['input_uap']['fr']:.4f}, ASR={r['input_uap']['asr']:.4f}")


if __name__ == '__main__':
    import torchvision
    main()
