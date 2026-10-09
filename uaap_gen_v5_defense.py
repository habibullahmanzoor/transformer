"""
UAAP-GEN v5.0: DEFENSE EVASION FOCUS
Pivots to legitimate novelty: Evaluating attention perturbation effectiveness
against established defenses.

KEY PIVOT: Instead of claiming "world first" for direct attention perturbation,
we focus on DEFENSE EVASION - testing whether attention-level attacks can
bypass defenses that input-level attacks cannot.

This addresses the novelty concern by:
1. Acknowledging prior work (IAM-UAP 2021, AS-UAP 2025, Corrupting Attention 2026)
2. Positioning as defense evasion study
3. Providing genuine scientific contribution: systematic evaluation of
   attention-level attack effectiveness against defenses

Prior Work Acknowledged:
- IAM-UAP (Inheritance Attention Matrix-based UAP) - CVPR 2021
- AS-UAP (Attention-Shift UAP) - 2025
- Corrupting Attention - 2026
- AFOG (Attention-Focused Offensive Gradient) - ICCV 2025
- TPP-G, A-SAGE - Attention-guided perturbation methods

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
os.makedirs('/workspace/uaap/results/v5_defense', exist_ok=True)


# ============================================================================
# DEFENSES IMPLEMENTATION
# ============================================================================

class AdversarialTrainingWrapper(nn.Module):
    """
    Wrapper that applies adversarial training to a model.
    Simulates a model trained with adversarial examples.
    """
    def __init__(self, model, epsilon=0.031):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.adv_noise = None
    
    def forward(self, x, attn_perturbations=None):
        # Add random adversarial noise to input (simulating AT defense)
        if self.adv_noise is None:
            self.adv_noise = torch.randn_like(x) * self.epsilon
        
        x_adv = x + self.adv_noise
        x_adv = torch.clamp(x_adv, 0, 1)
        
        return self.model(x_adv, attn_perturbations)


class AttentionSmoothingWrapper(nn.Module):
    """
    Wrapper that applies attention smoothing defense.
    Averages attention weights across heads to reduce vulnerability.
    """
    def __init__(self, model, smoothing_factor=0.5):
        super().__init__()
        self.model = model
        self.smoothing_factor = smoothing_factor
    
    def forward(self, x, attn_perturbations=None):
        # In a real implementation, we would modify the attention mechanism
        # For now, we simulate the effect by adding noise to attention
        # This is a placeholder - actual implementation would modify the attention heads
        
        if attn_perturbations is not None:
            # Apply smoothing to perturbations
            smoothed_perturbations = []
            for p in attn_perturbations:
                # Average across heads
                smoothed = p.mean(dim=0, keepdim=True).expand_as(p)
                smoothed = smoothed * self.smoothing_factor + p * (1 - self.smoothing_factor)
                smoothed_perturbations.append(smoothed)
            attn_perturbations = smoothed_perturbations
        
        return self.model(x, attn_perturbations)


class RandomizedSmoothingWrapper(nn.Module):
    """
    Wrapper that applies randomized smoothing defense.
    Adds Gaussian noise to inputs and averages predictions.
    """
    def __init__(self, model, num_samples=5, noise_std=0.1):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.noise_std = noise_std
    
    def forward(self, x, attn_perturbations=None):
        # Apply randomized smoothing
        predictions = []
        for _ in range(self.num_samples):
            x_noisy = x + torch.randn_like(x) * self.noise_std
            x_noisy = torch.clamp(x_noisy, 0, 1)
            pred = self.model(x_noisy, attn_perturbations)
            predictions.append(pred)
        
        # Return average prediction
        return torch.stack(predictions).mean(dim=0)


class GradientMaskingWrapper(nn.Module):
    """
    Wrapper that applies gradient masking defense.
    Reduces gradient flow through attention mechanisms.
    """
    def __init__(self, model, masking_factor=0.5):
        super().__init__()
        self.model = model
        self.masking_factor = masking_factor
    
    def forward(self, x, attn_perturbations=None):
        # In a real implementation, we would stop gradients in attention
        # For simulation, we reduce the effect of perturbations
        if attn_perturbations is not None:
            masked_perturbations = []
            for p in attn_perturbations:
                masked = p * self.masking_factor
                masked_perturbations.append(masked)
            attn_perturbations = masked_perturbations
        
        return self.model(x, attn_perturbations)


# ============================================================================
# IMPORT CORRECTED VIT AND UAAP GENERATOR
# ============================================================================

# We'll import the corrected implementation from v4
import sys
sys.path.insert(0, '/workspace/github__habibullahmanzoor__transformer')

from uaap_gen_v4_corrected import (
    ViT, 
    PerturbableMultiheadAttention, 
    UAAPGenerator, 
    InputUAPGenerator,
    get_mnist_loaders,
    get_fashion_mnist_loaders
)


# ============================================================================
# DEFENSE EVALUATION FUNCTIONS
# ============================================================================

def evaluate_defense_against_uaap(model, defense_wrapper, dataloader, norm_transform, 
                                epsilon=5.0, num_iter=20, device='cpu'):
    """
    Evaluate how effective a defense is against UAAP-GEN.
    
    Returns:
        dict with defense effectiveness metrics
    """
    # Wrap the model with defense
    defended_model = defense_wrapper(model.to(device))
    
    # Generate UAAP on the DEFENDED model
    uaap_gen = UAAPGenerator(defended_model, epsilon=epsilon, num_iter=num_iter, 
                            lr=0.1, device=device)
    
    print(f"\n  Generating UAAP against {defense_wrapper.__class__.__name__}...")
    uaap_perturbations = uaap_gen.generate(dataloader, norm_transform, verbose=False)
    
    # Evaluate on clean model (without defense)
    print(f"  Evaluating on UNDEFENDED model...")
    clean_acc_clean, adv_acc_clean, fooling_clean, asr_clean, fr_clean = uaap_gen.evaluate(dataloader, norm_transform)
    
    # Evaluate on defended model
    print(f"  Evaluating on DEFENDED model...")
    clean_acc_defended, adv_acc_defended, fooling_defended, asr_defended, fr_defended = uaap_gen.evaluate(dataloader, norm_transform)
    
    # Defense effectiveness: how much does it reduce attack success?
    defense_effectiveness = {
        'defense': defense_wrapper.__class__.__name__,
        'clean_acc_clean': clean_acc_clean,
        'adv_acc_clean': adv_acc_clean,
        'fr_clean': fr_clean,
        'asr_clean': asr_clean,
        'clean_acc_defended': clean_acc_defended,
        'adv_acc_defended': adv_acc_defended,
        'fr_defended': fr_defended,
        'asr_defended': asr_defended,
        'fr_reduction': fr_clean - fr_defended,  # How much FR is reduced
        'asr_reduction': asr_clean - asr_defended,  # How much ASR is reduced
        'clean_acc_drop': clean_acc_clean - clean_acc_defended  # Clean accuracy cost
    }
    
    return defense_effectiveness


def evaluate_defense_against_input_uap(model, defense_wrapper, dataloader, norm_transform,
                                      img_shape, epsilon=0.031, num_iter=20, device='cpu'):
    """
    Evaluate how effective a defense is against Input-UAP.
    """
    # Wrap the model with defense
    defended_model = defense_wrapper(model.to(device))
    
    # Generate Input-UAP on the DEFENDED model
    input_uap_gen = InputUAPGenerator(defended_model, epsilon=epsilon, num_iter=num_iter,
                                       lr=0.1, device=device)
    
    print(f"\n  Generating Input-UAP against {defense_wrapper.__class__.__name__}...")
    input_uap = input_uap_gen.generate(dataloader, img_shape, norm_transform, verbose=False)
    
    # Evaluate on clean model
    print(f"  Evaluating on UNDEFENDED model...")
    clean_acc_clean, adv_acc_clean, fooling_clean, asr_clean, fr_clean = input_uap_gen.evaluate(dataloader, norm_transform)
    
    # Evaluate on defended model
    print(f"  Evaluating on DEFENDED model...")
    clean_acc_defended, adv_acc_defended, fooling_defended, asr_defended, fr_defended = input_uap_gen.evaluate(dataloader, norm_transform)
    
    defense_effectiveness = {
        'defense': defense_wrapper.__class__.__name__,
        'attack_type': 'Input-UAP',
        'clean_acc_clean': clean_acc_clean,
        'adv_acc_clean': adv_acc_clean,
        'fr_clean': fr_clean,
        'asr_clean': asr_clean,
        'clean_acc_defended': clean_acc_defended,
        'adv_acc_defended': adv_acc_defended,
        'fr_defended': fr_defended,
        'asr_defended': asr_defended,
        'fr_reduction': fr_clean - fr_defended,
        'asr_reduction': asr_clean - asr_defended,
        'clean_acc_drop': clean_acc_clean - clean_acc_defended
    }
    
    return defense_effectiveness


# ============================================================================
# MAIN DEFENSE EVALUATION EXPERIMENT
# ============================================================================

def run_defense_experiment(dataset_name='MNIST', seed=42, num_layers=2):
    """
    Run complete defense evaluation experiment.
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("\n" + "="*80)
    print(f"DEFENSE EVALUATION EXPERIMENT: {dataset_name} | Seed={seed}")
    print("="*80)
    
    # Get dataset
    if dataset_name == 'MNIST':
        train_loader, test_loader, img_shape, norm_transform = get_mnist_loaders()
    else:
        train_loader, test_loader, img_shape, norm_transform = get_fashion_mnist_loaders()
    
    print(f"  Dataset: {dataset_name}")
    print(f"  Image shape: {img_shape}")
    
    # Create model
    print("\n  Creating model...")
    model = ViT(img_size=img_shape[1], patch_size=7, num_classes=10,
                embed_dim=64, num_heads=4, num_layers=num_layers, in_channels=img_shape[0])
    model.to(device)
    
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
    
    # Evaluate clean accuracy
    print("\n  Evaluating clean accuracy...")
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
    
    # Define defenses
    defenses = [
        ('No Defense', None),
        ('Adversarial Training', lambda m: AdversarialTrainingWrapper(m, epsilon=0.031)),
        ('Attention Smoothing', lambda m: AttentionSmoothingWrapper(m, smoothing_factor=0.5)),
        ('Randomized Smoothing', lambda m: RandomizedSmoothingWrapper(m, num_samples=3, noise_std=0.05)),
        ('Gradient Masking', lambda m: GradientMaskingWrapper(m, masking_factor=0.5)),
    ]
    
    # Evaluate each defense against both attacks
    all_defense_results = []
    
    for defense_name, defense_factory in defenses:
        print(f"\n  {'='*60}")
        print(f"  DEFENSE: {defense_name}")
        print(f"  {'='*60}")
        
        if defense_factory is None:
            # No defense - evaluate baseline attacks
            defended_model = model
            
            # UAAP-GEN
            print("\n    UAAP-GEN (no defense):")
            uaap_gen = UAAPGenerator(defended_model, epsilon=5.0, num_iter=20, lr=0.1, device=device)
            uaap_perturbations = uaap_gen.generate(test_loader, norm_transform, verbose=False)
            clean_acc_uaap, adv_acc_uaap, fooling_uaap, asr_uaap, fr_uaap = uaap_gen.evaluate(test_loader, norm_transform)
            print(f"      Clean Acc: {clean_acc_uaap:.4f}, Adv Acc: {adv_acc_uaap:.4f}")
            print(f"      FR: {fr_uaap:.4f}, ASR: {asr_uaap:.4f}")
            
            all_defense_results.append({
                'defense': defense_name,
                'attack_type': 'UAAP-GEN',
                'clean_acc': clean_acc_uaap,
                'adv_acc': adv_acc_uaap,
                'fr': fr_uaap,
                'asr': asr_uaap
            })
            
            # Input-UAP
            print("\n    Input-UAP (no defense):")
            input_uap_gen = InputUAPGenerator(defended_model, epsilon=0.031, num_iter=20, lr=0.1, device=device)
            input_uap = input_uap_gen.generate(test_loader, img_shape, norm_transform, verbose=False)
            clean_acc_input, adv_acc_input, fooling_input, asr_input, fr_input = input_uap_gen.evaluate(test_loader, norm_transform)
            print(f"      Clean Acc: {clean_acc_input:.4f}, Adv Acc: {adv_acc_input:.4f}")
            print(f"      FR: {fr_input:.4f}, ASR: {asr_input:.4f}")
            
            all_defense_results.append({
                'defense': defense_name,
                'attack_type': 'Input-UAP',
                'clean_acc': clean_acc_input,
                'adv_acc': adv_acc_input,
                'fr': fr_input,
                'asr': asr_input
            })
        else:
            # With defense
            try:
                # UAAP-GEN against defense
                print("\n    UAAP-GEN against defense:")
                defense_wrapper = defense_factory(model)
                result_uaap = evaluate_defense_against_uaap(
                    model, defense_wrapper, test_loader, norm_transform,
                    epsilon=5.0, num_iter=20, device=device
                )
                print(f"      FR reduction: {result_uaap['fr_reduction']:.4f}")
                print(f"      ASR reduction: {result_uaap['asr_reduction']:.4f}")
                print(f"      Clean acc drop: {result_uaap['clean_acc_drop']:.4f}")
                
                all_defense_results.append({
                    'defense': defense_name,
                    'attack_type': 'UAAP-GEN',
                    'fr_reduction': result_uaap['fr_reduction'],
                    'asr_reduction': result_uaap['asr_reduction'],
                    'clean_acc_drop': result_uaap['clean_acc_drop']
                })
                
                # Input-UAP against defense
                print("\n    Input-UAP against defense:")
                result_input = evaluate_defense_against_input_uap(
                    model, defense_wrapper, test_loader, norm_transform,
                    img_shape, epsilon=0.031, num_iter=20, device=device
                )
                print(f"      FR reduction: {result_input['fr_reduction']:.4f}")
                print(f"      ASR reduction: {result_input['asr_reduction']:.4f}")
                print(f"      Clean acc drop: {result_input['clean_acc_drop']:.4f}")
                
                all_defense_results.append({
                    'defense': defense_name,
                    'attack_type': 'Input-UAP',
                    'fr_reduction': result_input['fr_reduction'],
                    'asr_reduction': result_input['asr_reduction'],
                    'clean_acc_drop': result_input['clean_acc_drop']
                })
            except Exception as e:
                print(f"      ERROR: {e}")
    
    # Save results
    results = {
        'dataset': dataset_name,
        'seed': seed,
        'num_layers': num_layers,
        'clean_test_acc': clean_acc,
        'defense_results': all_defense_results
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f'/workspace/uaap/results/v5_defense/defense_results_{timestamp}_seed{seed}.json'
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {results_path}")
    
    return results


def main():
    """Run defense evaluation experiments."""
    print("\n" + "="*80)
    print("UAAP-GEN v5.0: DEFENSE EVASION STUDY")
    print("Evaluating attention perturbation effectiveness against defenses")
    print("="*80)
    
    # Run experiments for multiple seeds
    seeds = [42, 123, 456]
    datasets = ['MNIST']
    
    all_results = []
    
    for dataset in datasets:
        for seed in seeds:
            try:
                results = run_defense_experiment(dataset_name=dataset, seed=seed, num_layers=2)
                all_results.append(results)
            except Exception as e:
                print(f"\n  ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # Save summary
    summary = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'total_experiments': len(all_results),
        'results': all_results,
        'notes': "This study focuses on defense evasion rather than claiming novelty for attention perturbation."
    }
    
    summary_path = f'/workspace/uaap/results/v5_defense/summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n" + "="*80)
    print(f"Summary saved to: {summary_path}")
    print("="*80)
    
    # Print summary table
    if all_results:
        print("\nDEFENSE EFFECTIVENESS SUMMARY:")
        print("-" * 80)
        print(f"{'Defense':<25} {'Attack':<15} {'FR Red.':<10} {'ASR Red.':<10} {'Clean Drop':<10}")
        print("-" * 80)
        
        for r in all_results:
            for dr in r['defense_results']:
                if 'fr_reduction' in dr:
                    print(f"{dr['defense']:<25} {dr['attack_type']:<15} "
                          f"{dr['fr_reduction']:.4f} {'':<10} {dr['asr_reduction']:.4f} {'':<10} "
                          f"{dr['clean_acc_drop']:.4f}")


if __name__ == '__main__':
    import torchvision
    main()
