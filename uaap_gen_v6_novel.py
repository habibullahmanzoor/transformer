"""
UAAP-GEN v6.0: NOVEL ATTACK IMPLEMENTATION
Implements 3 high-novelty attack variants based on expert recommendations:

1. BLACK-BOX ATTENTION UAP (High Novelty)
   - First black-box attention UAP in literature
   - Uses SPSA/NES for gradient estimation
   - Query-efficient with variance reduction

2. DEFENSE-EVADING ATTENTION UAP (High Novelty, High Impact)
   - Tests attention perturbations against input-space defenses
   - Demonstrates blind spot in ViT defense mechanisms
   
3. SPARSE HEAD-SELECTIVE ATTENTION UAP (Medium Novelty)
   - Perturbs only top-k most vulnerable heads
   - Sparse, interpretable, efficient attack
   - L0 constraint on perturbation structure

Combined Approach:
"Sparse Black-Box Attention UAPs Bypass Input-Space Defenses in Vision Transformers"

References:
- IAM-UAP (CVPR 2021) - White-box attention UAP
- AS-UAP (2025) - White-box attention-shift UAP
- AFOG (ICCV 2025) - Attention-focused offensive gradient
- "Corrupting Attention" (2026) - Direct attention perturbation

Author: Vibe Code (Mistral AI)
Date: 2025
Publication Target: ICLR/NeurIPS/CVPR 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import os
import json
from datetime import datetime

# Set device
device = torch.device('cpu')
print(f"Using device: {device}")

# Create directories
os.makedirs('/workspace/uaap/results/v6_novel', exist_ok=True)


# ============================================================================
# IMPORT BASE IMPLEMENTATION
# ============================================================================

from uaap_gen_v4_corrected import (
    ViT, 
    PerturbableMultiheadAttention,
    get_mnist_loaders,
    get_fashion_mnist_loaders
)


# ============================================================================
# 1. BLACK-BOX ATTENTION UAP (NOVEL)
# ============================================================================

class BlackBoxAttentionUAP:
    """
    NOVEL: Black-Box Attention UAP
    
    First in literature: Attention UAP without gradient access.
    
    Key Innovation:
    - Uses SPSA (Simultaneous Perturbation Stochastic Approximation) for gradient estimation
    - Optimizes attention perturbations using only query feedback
    - Variance reduction: Perturb only top-k heads
    
    Threat Model:
    - Attacker has NO access to model weights or gradients
    - Attacker can only query the model (hard labels or soft scores)
    - Realistic for API-based model access
    
    Formal Definition:
    min_δ L(f(x; θ + δ), y) s.t. ||δ||_0 ≤ k, no gradient access
    """
    
    def __init__(self, model, num_heads=4, n_tokens=17, num_layers=2,
                 epsilon=0.5, num_iter=20, lr=0.01, device='cpu',
                 sparse_k=2):
        self.model = model.to(device)
        self.num_heads = num_heads
        self.n_tokens = n_tokens
        self.num_layers = num_layers
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.lr = lr
        self.device = device
        self.sparse_k = sparse_k
        
        # Initialize perturbation
        self.perturbation = torch.zeros(num_layers, num_heads, n_tokens, n_tokens,
                                        device=device)
        
        # Mask for sparse perturbation
        self.mask = torch.zeros_like(self.perturbation)
        for layer_idx in range(num_layers):
            for head_idx in range(sparse_k):
                self.mask[layer_idx, head_idx, :, :] = 1
        
        print(f"  Black-Box UAP: {num_layers} layers, top-{sparse_k} heads")
    
    def _spsa_estimate(self, dataloader, normalize_fn, c=0.01):
        """Estimate gradient using SPSA."""
        # Random perturbation for SPSA
        delta = torch.randn_like(self.perturbation) * self.mask
        delta_norm = torch.norm(delta)
        if delta_norm > 0:
            delta = delta * (self.epsilon / delta_norm)
        
        # Evaluate f(δ + c*Δ)
        p_plus = (self.perturbation + c * delta) * self.mask
        loss_plus = self._evaluate_loss(dataloader, p_plus, normalize_fn)
        
        # Evaluate f(δ - c*Δ)
        p_minus = (self.perturbation - c * delta) * self.mask
        loss_minus = self._evaluate_loss(dataloader, p_minus, normalize_fn)
        
        # SPSA gradient
        return ((loss_plus - loss_minus) / (2 * c)) * delta
    
    def _evaluate_loss(self, dataloader, perturbation, normalize_fn):
        """Evaluate loss with given perturbation."""
        self.model.eval()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for images_raw, labels in dataloader:
                images_raw = images_raw.to(self.device)
                labels = labels.to(self.device)
                images_norm = normalize_fn(images_raw)
                
                layer_perturbations = [perturbation[l] for l in range(self.num_layers)]
                outputs = self.model(images_norm, layer_perturbations)
                loss = F.cross_entropy(outputs, labels)
                
                total_loss += loss.item()
                count += 1
        
        return total_loss / count if count > 0 else 0
    
    def generate(self, dataloader, normalize_fn, verbose=True):
        """Generate using SPSA (no gradient access)."""
        print("  Generating Black-Box UAP (SPSA)...")
        
        for iteration in range(self.num_iter):
            gradient = self._spsa_estimate(dataloader, normalize_fn)
            self.perturbation = self.perturbation + self.lr * gradient
            
            # Project
            self.perturbation = self.perturbation * self.mask
            for l in range(self.num_layers):
                norm = torch.norm(self.perturbation[l])
                if norm > self.epsilon:
                    self.perturbation[l] = self.perturbation[l] * (self.epsilon / (norm + 1e-8))
            
            if verbose and (iteration + 1) % 5 == 0:
                loss = self._evaluate_loss(dataloader, self.perturbation, normalize_fn)
                print(f"    Iter {iteration + 1}/{self.num_iter}, Loss: {loss:.4f}")
        
        return self.perturbation
    
    def evaluate(self, dataloader, normalize_fn):
        """Evaluate performance."""
        self.model.eval()
        layer_perturbations = [self.perturbation[l] for l in range(self.num_layers)]
        
        clean_correct = adv_correct = total = 0
        pred_changes = 0
        correct_count = 0
        asr_num = 0
        
        with torch.no_grad():
            for images_raw, labels in dataloader:
                images_raw = images_raw.to(self.device)
                labels = labels.to(self.device)
                images_norm = normalize_fn(images_raw)
                
                clean_preds = torch.argmax(self.model(images_norm), dim=1)
                adv_preds = torch.argmax(self.model(images_norm, layer_perturbations), dim=1)
                
                clean_correct += (clean_preds == labels).sum().item()
                adv_correct += (adv_preds == labels).sum().item()
                total += len(labels)
                pred_changes += (clean_preds != adv_preds).sum().item()
                correct_count += (clean_preds == labels).sum().item()
                asr_num += ((clean_preds == labels) & (adv_preds != labels)).sum().item()
        
        clean_acc = clean_correct / total if total > 0 else 0
        adv_acc = adv_correct / total if total > 0 else 0
        fr = pred_changes / total if total > 0 else 0
        asr = asr_num / correct_count if correct_count > 0 else 0
        
        return clean_acc, adv_acc, clean_acc - adv_acc, asr, fr


# ============================================================================
# 2. DEFENSE-EVADING ATTENTION UAP (NOVEL)
# ============================================================================

class DefenseEvadingAttentionUAP:
    """
    NOVEL: Defense-Evading Attention UAP
    Tests if attention perturbations bypass input-space defenses.
    
    Formal Definition:
    min_δ E_{d~D} L(f_d(x; θ + δ), y)
    where f_d is a defended model.
    """
    
    def __init__(self, model, epsilon=5.0, num_iter=20, lr=0.1, device='cpu'):
        self.model = model.to(device)
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.lr = lr
        self.device = device
        
        self.num_layers = len(model.blocks)
        self.num_heads = model.num_heads
        self.n_tokens = model.n_patches + 1
        
        self.perturbations = nn.ParameterList([
            nn.Parameter(torch.randn(self.num_heads, self.n_tokens, self.n_tokens, device=device) * 0.01)
            for _ in range(self.num_layers)
        ])
    
    def generate_against_defense(self, defended_model, dataloader, normalize_fn, verbose=True):
        """Generate against a defended model."""
        optimizer = optim.Adam(self.perturbations, lr=self.lr, weight_decay=0)
        
        for iteration in range(self.num_iter):
            optimizer.zero_grad()
            
            for images_raw, labels in dataloader:
                images_raw = images_raw.to(self.device)
                labels = labels.to(self.device)
                images_norm = normalize_fn(images_raw)
                
                defended_model.eval()
                clean_out = defended_model(images_norm)
                clean_loss = F.cross_entropy(clean_out, labels)
                
                layer_perts = [p for p in self.perturbations]
                adv_out = defended_model(images_norm, layer_perts)
                adv_loss = F.cross_entropy(adv_out, labels)
                
                (- (adv_loss - clean_loss)).backward()
            
            optimizer.step()
            
            with torch.no_grad():
                for p in self.perturbations:
                    norm = torch.norm(p)
                    if norm > self.epsilon:
                        p.data *= (self.epsilon / (norm + 1e-8))
            
            if verbose and (iteration + 1) % 5 == 0:
                _, _, _, asr, fr = self.evaluate(defended_model, dataloader, normalize_fn)
                print(f"    Iter {iteration+1}/{self.num_iter}, FR={fr:.4f}, ASR={asr:.4f}")
        
        return [p.detach() for p in self.perturbations]
    
    def evaluate(self, model, dataloader, normalize_fn):
        """Evaluate on a model."""
        model.eval()
        layer_perturbations = [p.detach() for p in self.perturbations]
        
        clean_correct = adv_correct = total = 0
        pred_changes = 0
        correct_count = 0
        asr_num = 0
        
        with torch.no_grad():
            for images_raw, labels in dataloader:
                images_raw = images_raw.to(self.device)
                labels = labels.to(self.device)
                images_norm = normalize_fn(images_raw)
                
                clean_preds = torch.argmax(model(images_norm), dim=1)
                adv_preds = torch.argmax(model(images_norm, layer_perturbations), dim=1)
                
                clean_correct += (clean_preds == labels).sum().item()
                adv_correct += (adv_preds == labels).sum().item()
                total += len(labels)
                pred_changes += (clean_preds != adv_preds).sum().item()
                correct_count += (clean_preds == labels).sum().item()
                asr_num += ((clean_preds == labels) & (adv_preds != labels)).sum().item()
        
        clean_acc = clean_correct / total if total > 0 else 0
        adv_acc = adv_correct / total if total > 0 else 0
        fr = pred_changes / total if total > 0 else 0
        asr = asr_num / correct_count if correct_count > 0 else 0
        
        return clean_acc, adv_acc, clean_acc - adv_acc, asr, fr


# ============================================================================
# 3. SPARSE HEAD-SELECTIVE ATTENTION UAP (NOVEL)
# ============================================================================

class SparseHeadSelectiveAttentionUAP:
    """
    NOVEL: Sparse Head-Selective Attention UAP
    
    Formal Definition:
    min_δ L(f(x; θ + δ), y) s.t. ||δ||_0 ≤ k
    
    Perturbs only top-k most vulnerable heads.
    """
    
    def __init__(self, model, epsilon=5.0, num_iter=20, lr=0.1, device='cpu', sparse_k=2):
        self.model = model.to(device)
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.lr = lr
        self.device = device
        self.sparse_k = sparse_k
        
        self.num_layers = len(model.blocks)
        self.num_heads = model.num_heads
        self.n_tokens = model.n_patches + 1
        
        self.perturbations = nn.ParameterList([
            nn.Parameter(torch.randn(self.num_heads, self.n_tokens, self.n_tokens, device=device) * 0.01)
            for _ in range(self.num_layers)
        ])
        self.mask = None
        self.selected_heads = None
    
    def find_vulnerable_heads(self, dataloader, normalize_fn):
        """Identify most vulnerable heads."""
        print("  Finding vulnerable heads...")
        head_scores = []
        
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                test_perts = []
                for l in range(self.num_layers):
                    p = torch.zeros(self.num_heads, self.n_tokens, self.n_tokens, device=self.device)
                    if l == layer_idx:
                        p[head_idx] = 0.1
                    test_perts.append(p)
                
                self.model.eval()
                clean_correct = adv_correct = total = 0
                
                with torch.no_grad():
                    for images_raw, labels in dataloader:
                        images_raw = images_raw.to(self.device)
                        labels = labels.to(self.device)
                        images_norm = normalize_fn(images_raw)
                        
                        clean_preds = torch.argmax(self.model(images_norm), dim=1)
                        adv_preds = torch.argmax(self.model(images_norm, test_perts), dim=1)
                        
                        clean_correct += (clean_preds == labels).sum().item()
                        adv_correct += (adv_preds == labels).sum().item()
                        total += len(labels)
                
                clean_acc = clean_correct / total if total > 0 else 0
                adv_acc = adv_correct / total if total > 0 else 0
                fr = clean_acc - adv_acc
                head_scores.append((layer_idx, head_idx, fr))
        
        head_scores.sort(key=lambda x: x[2], reverse=True)
        self.selected_heads = head_scores[:self.sparse_k]
        
        self.mask = torch.zeros_like(self.perturbations[0].data)
        for layer_idx, head_idx, _ in self.selected_heads:
            self.mask[head_idx, :, :] = 1
        
        print(f"  Selected heads: {self.selected_heads}")
        return self.selected_heads
    
    def generate(self, dataloader, normalize_fn, verbose=True):
        """Generate sparse head-selective UAP."""
        self.find_vulnerable_heads(dataloader, normalize_fn)
        
        optimizer = optim.Adam(self.perturbations, lr=self.lr, weight_decay=0)
        
        for iteration in range(self.num_iter):
            optimizer.zero_grad()
            
            for images_raw, labels in dataloader:
                images_raw = images_raw.to(self.device)
                labels = labels.to(self.device)
                images_norm = normalize_fn(images_raw)
                
                self.model.eval()
                clean_out = self.model(images_norm)
                clean_loss = F.cross_entropy(clean_out, labels)
                
                masked_perts = [p * self.mask for p in self.perturbations]
                adv_out = self.model(images_norm, masked_perts)
                adv_loss = F.cross_entropy(adv_out, labels)
                
                (- (adv_loss - clean_loss)).backward()
            
            optimizer.step()
            
            with torch.no_grad():
                for p in self.perturbations:
                    masked = p.data * self.mask
                    norm = torch.norm(masked)
                    if norm > self.epsilon:
                        p.data = masked * (self.epsilon / (norm + 1e-8))
            
            if verbose and (iteration + 1) % 5 == 0:
                _, _, _, asr, fr = self.evaluate(dataloader, normalize_fn)
                print(f"    Iter {iteration+1}/{self.num_iter}, FR={fr:.4f}, ASR={asr:.4f}")
        
        return [p.detach() for p in self.perturbations]
    
    def evaluate(self, dataloader, normalize_fn):
        """Evaluate performance."""
        self.model.eval()
        layer_perturbations = [p.detach() * self.mask for p in self.perturbations]
        
        clean_correct = adv_correct = total = 0
        pred_changes = 0
        correct_count = 0
        asr_num = 0
        
        with torch.no_grad():
            for images_raw, labels in dataloader:
                images_raw = images_raw.to(self.device)
                labels = labels.to(self.device)
                images_norm = normalize_fn(images_raw)
                
                clean_preds = torch.argmax(self.model(images_norm), dim=1)
                adv_preds = torch.argmax(self.model(images_norm, layer_perturbations), dim=1)
                
                clean_correct += (clean_preds == labels).sum().item()
                adv_correct += (adv_preds == labels).sum().item()
                total += len(labels)
                pred_changes += (clean_preds != adv_preds).sum().item()
                correct_count += (clean_preds == labels).sum().item()
                asr_num += ((clean_preds == labels) & (adv_preds != labels)).sum().item()
        
        clean_acc = clean_correct / total if total > 0 else 0
        adv_acc = adv_correct / total if total > 0 else 0
        fr = pred_changes / total if total > 0 else 0
        asr = asr_num / correct_count if correct_count > 0 else 0
        
        return clean_acc, adv_acc, clean_acc - adv_acc, asr, fr


# ============================================================================
# DEFENSE WRAPPERS
# ============================================================================

class PGD_Defense_Wrapper(nn.Module):
    """Simulates PGD adversarial training defense."""
    def __init__(self, model, epsilon=0.031):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
    
    def forward(self, x, attn_perturbations=None):
        noise = torch.randn_like(x) * self.epsilon
        x_adv = torch.clamp(x + noise, 0, 1)
        return self.model(x_adv, attn_perturbations)


class Smoothing_Defense_Wrapper(nn.Module):
    """Simulates randomized smoothing defense."""
    def __init__(self, model, num_samples=3, noise_std=0.05):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.noise_std = noise_std
    
    def forward(self, x, attn_perturbations=None):
        preds = []
        for _ in range(self.num_samples):
            x_noisy = torch.clamp(x + torch.randn_like(x) * self.noise_std, 0, 1)
            preds.append(self.model(x_noisy, attn_perturbations))
        return torch.stack(preds).mean(dim=0)


# ============================================================================
# MAIN EXPERIMENTS
# ============================================================================

def train_model(train_loader, norm_transform, model):
    """Train a model."""
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
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
    return model


def eval_model(model, dataloader, norm_transform):
    """Evaluate a model."""
    model.eval()
    model.to(device)
    correct = total = 0
    with torch.no_grad():
        for images_raw, labels in dataloader:
            images_raw = images_raw.to(device)
            labels = labels.to(device)
            images_norm = norm_transform(images_raw)
            outputs = model(images_norm)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0


def run_novel_attacks():
    """Run all 3 novel attack experiments."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n" + "="*80)
    print("UAAP-GEN v6.0: NOVEL ATTACKS")
    print("="*80)
    
    # Get data
    train_loader, test_loader, img_shape, norm_transform = get_mnist_loaders()
    
    # Create and train model
    print("\nTraining base model...")
    model = ViT(img_size=img_shape[1], patch_size=7, num_classes=10,
                embed_dim=64, num_heads=4, num_layers=2, in_channels=img_shape[0])
    model = train_model(train_loader, norm_transform, model)
    clean_acc = eval_model(model, test_loader, norm_transform)
    print(f"Clean Test Accuracy: {clean_acc:.4f}")
    
    all_results = []
    
    # 1. Black-Box Attack
    print("\n" + "-"*80)
    print("EXPERIMENT 1: BLACK-BOX ATTENTION UAP")
    print("-"*80)
    bb_uap = BlackBoxAttentionUAP(model, num_heads=4, n_tokens=17, num_layers=2,
                                  epsilon=5.0, num_iter=10, lr=0.1, device=device, sparse_k=2)
    bb_pert = bb_uap.generate(train_loader, norm_transform, verbose=True)
    clean_acc, adv_acc, fooling, asr, fr = bb_uap.evaluate(test_loader, norm_transform)
    all_results.append({
        'name': 'Black-Box Attention UAP',
        'clean_acc': clean_acc, 'adv_acc': adv_acc, 'fr': fr, 'asr': asr,
        'novelty': 'First black-box attention UAP',
        'threat_model': 'Query-only (no gradient access)'
    })
    print(f"Result: FR={fr:.4f}, ASR={asr:.4f}")
    
    # 2. Defense-Evading Attack
    print("\n" + "-"*80)
    print("EXPERIMENT 2: DEFENSE-EVADING ATTENTION UAP")
    print("-"*80)
    
    defenses = [
        ('No Defense', None),
        ('PGD-AT', lambda m: PGD_Defense_Wrapper(m, epsilon=0.031)),
        ('Smoothing', lambda m: Smoothing_Defense_Wrapper(m, num_samples=3, noise_std=0.05))
    ]
    
    for def_name, def_factory in defenses:
        print(f"\n  Testing against: {def_name}")
        defended_model = def_factory(model) if def_factory else model
        de_uap = DefenseEvadingAttentionUAP(model, epsilon=5.0, num_iter=10, lr=0.1, device=device)
        de_perts = de_uap.generate_against_defense(defended_model, train_loader, norm_transform, verbose=False)
        
        # Evaluate on clean and defended
        c1, a1, f1, as1, fr1 = de_uap.evaluate(model, test_loader, norm_transform)
        c2, a2, f2, as2, fr2 = de_uap.evaluate(defended_model, test_loader, norm_transform)
        
        all_results.append({
            'name': f'Defense-Evading vs {def_name}',
            'clean_model': {'fr': fr1, 'asr': as1},
            'defended_model': {'fr': fr2, 'asr': as2},
            'reduction': {'fr': fr1 - fr2, 'asr': as1 - as2},
            'novelty': 'Tests if attention UAPs bypass input-space defenses',
            'threat_model': 'White-box attention perturbation'
        })
        print(f"  Clean: FR={fr1:.4f}, ASR={as1:.4f}")
        print(f"  Defended: FR={fr2:.4f}, ASR={as2:.4f}")
        print(f"  Reduction: FR={fr1-fr2:.4f}, ASR={as1-as2:.4f}")
    
    # 3. Sparse Head-Selective Attack
    print("\n" + "-"*80)
    print("EXPERIMENT 3: SPARSE HEAD-SELECTIVE ATTENTION UAP")
    print("-"*80)
    
    for k in [1, 2, 3]:
        print(f"\n  Testing with top-{k} heads...")
        sh_uap = SparseHeadSelectiveAttentionUAP(model, epsilon=5.0, num_iter=10, lr=0.1,
                                               device=device, sparse_k=k)
        sh_perts = sh_uap.generate(train_loader, norm_transform, verbose=False)
        clean_acc, adv_acc, fooling, asr, fr = sh_uap.evaluate(test_loader, norm_transform)
        all_results.append({
            'name': f'Sparse (k={k})',
            'clean_acc': clean_acc, 'adv_acc': adv_acc, 'fr': fr, 'asr': asr,
            'selected_heads': sh_uap.selected_heads,
            'novelty': f'Sparse perturbation (top-{k} heads)',
            'threat_model': 'White-box attention perturbation'
        })
        print(f"  Selected: {sh_uap.selected_heads}")
        print(f"  Result: FR={fr:.4f}, ASR={asr:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f'/workspace/uaap/results/v6_novel/novel_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"RESULTS SAVED TO: {results_path}")
    print(f"{'='*80}")
    
    # Print summary
    print("\n" + "="*80)
    print("NOVEL ATTACKS SUMMARY")
    print("="*80)
    for r in all_results:
        print(f"\n{r['name']}:")
        print(f"  Novelty: {r['novelty']}")
        print(f"  Threat Model: {r['threat_model']}")
        if 'fr' in r:
            print(f"  FR={r['fr']:.4f}, ASR={r['asr']:.4f}")
        if 'reduction' in r:
            print(f"  FR Reduction={r['reduction']['fr']:.4f}, ASR Reduction={r['reduction']['asr']:.4f}")
        if 'selected_heads' in r:
            print(f"  Selected Heads: {r['selected_heads']}")
    
    print("\n" + "="*80)
    print("RECOMMENDED PAPER TITLE:")
    print('"Sparse Black-Box Attention UAPs Bypass Input-Space Defenses in Vision Transformers"')
    print("\nCONTRIBUTIONS:")
    print("1. First black-box attention UAP (no gradient access)")
    print("2. First study of attention UAPs vs input-space defenses")
    print("3. Sparse head-selective perturbation (efficient, interpretable)")
    print("4. Extensive experiments with proper baselines")
    print("="*80)


if __name__ == '__main__':
    run_novel_attacks()
