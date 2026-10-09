"""
UAAP-GEN: Full Evaluation Pipeline
Runs complete evaluation with all baselines, datasets, and ablation studies.

This script orchestrates:
1. SOTA baseline comparison (IAM-UAP, AS-UAP, Input-UAP)
2. Scaled evaluation (CIFAR-10, CIFAR-100)
3. Ablation studies (layer-wise, head-wise, budget)
4. Defense evasion evaluation

Author: Vibe Code (Mistral AI)
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import json
import os

# Set device
device = torch.device('cpu')
print(f"Using device: {device}")

# Create output directory
os.makedirs('/workspace/uaap/results/full_evaluation', exist_ok=True)


# ============================================================================
# IMPORT ALL MODULES
# ============================================================================

# Import from our corrected implementation
from uaap_gen_v4_corrected import (
    ViT, 
    UAAPGenerator, 
    InputUAPGenerator,
    get_mnist_loaders,
    get_fashion_mnist_loaders
)

# Import baselines
from baselines_iam_as import IAM_UAP_Generator, AS_UAP_Generator, run_sota_comparison

# Import CIFAR support
from cifar10_deit_support import (
    DeiT_Tiny,
    get_cifar10_loaders,
    get_cifar100_loaders,
    run_cifar10_experiment,
    run_ablation_study
)

# Import defense study
from uaap_gen_v5_defense import (
    AdversarialTrainingWrapper,
    AttentionSmoothingWrapper,
    RandomizedSmoothingWrapper,
    GradientMaskingWrapper
)


# ============================================================================
# PHASE 3: SOTA BASELINE COMPARISON
# ============================================================================

def phase3_sota_comparison():
    """Run Phase 3: SOTA baseline comparison."""
    print("\n" + "="*80)
    print("PHASE 3: SOTA BASELINE COMPARISON")
    print("="*80)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Get MNIST data
    train_loader, test_loader, img_shape, norm_transform = get_mnist_loaders()
    
    # Create and train model
    print("\n  Creating and training ViT model...")
    model = ViT(img_size=img_shape[1], patch_size=7, num_classes=10,
                embed_dim=64, num_heads=4, num_layers=2, in_channels=img_shape[0])
    model.to(device)
    
    # Train
    model.train()
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
    
    # Run comparison
    print("\n  Running SOTA comparison...")
    results = run_sota_comparison(
        model, train_loader, test_loader, img_shape, norm_transform,
        epsilon=0.031, num_iter=20, device=device
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f'/workspace/uaap/results/full_evaluation/phase3_sota_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {results_path}")
    
    # Print summary
    print("\n  " + "="*80)
    print("  SOTA COMPARISON SUMMARY")
    print("  " + "="*80)
    print(f"  {'Method':<20} {'Clean Acc':<12} {'Adv Acc':<12} {'FR':<10} {'ASR':<10}")
    print("  " + "-" * 80)
    
    for method, metrics in results.items():
        print(f"  {method:<20} {metrics['clean_acc']:<12.4f} {metrics['adv_acc']:<12.4f} "
              f"{metrics['fr']:<10.4f} {metrics['asr']:<10.4f}")
    
    return results


# ============================================================================
# PHASE 4: SCALED EVALUATION
# ============================================================================

def phase4_scaled_evaluation():
    """Run Phase 4: Scaled evaluation on CIFAR-10/100."""
    print("\n" + "="*80)
    print("PHASE 4: SCALED EVALUATION")
    print("="*80)
    
    all_results = []
    
    # CIFAR-10
    print("\n  Running CIFAR-10 experiment...")
    try:
        cifar10_results = run_cifar10_experiment()
        all_results.append(cifar10_results)
        print(f"\n  CIFAR-10: Clean Acc = {cifar10_results['clean_acc']:.4f}")
        if 'uaap_gen' in cifar10_results:
            print(f"  UAAP-GEN: FR = {cifar10_results['uaap_gen']['fr']:.4f}, "
                  f"ASR = {cifar10_results['uaap_gen']['asr']:.4f}")
    except Exception as e:
        print(f"\n  CIFAR-10 ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # CIFAR-100
    print("\n  Running CIFAR-100 experiment...")
    try:
        cifar100_results = run_cifar100_experiment()
        all_results.append(cifar100_results)
        print(f"\n  CIFAR-100: Clean Acc = {cifar100_results['clean_acc']:.4f}")
    except Exception as e:
        print(f"\n  CIFAR-100 ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f'/workspace/uaap/results/full_evaluation/phase4_scaled_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n  Results saved to: {results_path}")
    
    return all_results


# ============================================================================
# PHASE 5: ABLATION STUDIES
# ============================================================================

def phase5_ablation_studies():
    """Run Phase 5: Ablation studies."""
    print("\n" + "="*80)
    print("PHASE 5: ABLATION STUDIES")
    print("="*80)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Get MNIST data
    train_loader, test_loader, img_shape, norm_transform = get_mnist_loaders()
    
    # Create and train model
    print("\n  Creating and training model...")
    model = ViT(img_size=img_shape[1], patch_size=7, num_classes=10,
                embed_dim=64, num_heads=4, num_layers=2, in_channels=img_shape[0])
    model.to(device)
    
    # Train
    model.train()
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
    
    # Run ablation studies
    print("\n  Running ablation studies...")
    ablation_results = run_ablation_study(model, train_loader, test_loader,
                                          norm_transform, device)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f'/workspace/uaap/results/full_evaluation/phase5_ablation_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    print(f"\n  Results saved to: {results_path}")
    
    # Print summary
    print("\n  " + "="*80)
    print("  ABLATION STUDY SUMMARY")
    print("  " + "="*80)
    
    print("\n  Layer-wise:")
    for r in ablation_results.get('layer_wise', []):
        print(f"    {r['num_layers']} layers: FR={r['fr']:.4f}, ASR={r['asr']:.4f}")
    
    print("\n  Budget:")
    for r in ablation_results.get('budget', []):
        print(f"    epsilon={r['epsilon']}: FR={r['fr']:.4f}, ASR={r['asr']:.4f}")
    
    return ablation_results


# ============================================================================
# PHASE 6: DEFENSE EVALUATION
# ============================================================================

def phase6_defense_evaluation():
    """Run Phase 6: Defense evaluation."""
    print("\n" + "="*80)
    print("PHASE 6: DEFENSE EVALUATION")
    print("="*80)
    
    # Import defense evaluation functions
    from uaap_gen_v5_defense import run_defense_experiment
    
    all_results = []
    
    # Run for multiple seeds
    for seed in [42, 123, 456]:
        print(f"\n  Running defense experiment for seed {seed}...")
        try:
            results = run_defense_experiment(dataset_name='MNIST', seed=seed, num_layers=2)
            all_results.append(results)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f'/workspace/uaap/results/full_evaluation/phase6_defense_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n  Results saved to: {results_path}")
    
    return all_results


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def main():
    """Run complete evaluation pipeline."""
    print("\n" + "="*80)
    print("UAAP-GEN: FULL EVALUATION PIPELINE")
    print("Phases 3-6: SOTA Comparison, Scaled Evaluation, Ablation, Defense")
    print("="*80)
    
    all_phase_results = {}
    
    # Phase 3: SOTA Baseline Comparison
    print("\n" + "#"*80)
    print("# PHASE 3: SOTA BASELINE COMPARISON")
    print("#"*80)
    try:
        phase3_results = phase3_sota_comparison()
        all_phase_results['phase3'] = phase3_results
    except Exception as e:
        print(f"\nPhase 3 ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Phase 4: Scaled Evaluation
    print("\n" + "#"*80)
    print("# PHASE 4: SCALED EVALUATION (CIFAR-10/100)")
    print("#"*80)
    try:
        phase4_results = phase4_scaled_evaluation()
        all_phase_results['phase4'] = phase4_results
    except Exception as e:
        print(f"\nPhase 4 ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Phase 5: Ablation Studies
    print("\n" + "#"*80)
    print("# PHASE 5: ABLATION STUDIES")
    print("#"*80)
    try:
        phase5_results = phase5_ablation_studies()
        all_phase_results['phase5'] = phase5_results
    except Exception as e:
        print(f"\nPhase 5 ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Phase 6: Defense Evaluation
    print("\n" + "#"*80)
    print("# PHASE 6: DEFENSE EVALUATION")
    print("#"*80)
    try:
        phase6_results = phase6_defense_evaluation()
        all_phase_results['phase6'] = phase6_results
    except Exception as e:
        print(f"\nPhase 6 ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Save master summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    master_path = f'/workspace/uaap/results/full_evaluation/master_summary_{timestamp}.json'
    
    with open(master_path, 'w') as f:
        json.dump(all_phase_results, f, indent=2)
    
    print(f"\n" + "="*80)
    print("MASTER SUMMARY SAVED TO:")
    print(f"  {master_path}")
    print("="*80)
    
    # Print overall summary
    print("\n" + "="*80)
    print("EVALUATION PIPELINE COMPLETE")
    print("="*80)
    print("\nCompleted phases:")
    for phase in ['phase3', 'phase4', 'phase5', 'phase6']:
        status = "✅ COMPLETED" if phase in all_phase_results else "❌ FAILED"
        print(f"  Phase {phase[5:]}: {status}")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("  1. Analyze results")
    print("  2. Generate comparison figures")
    print("  3. Update README with findings")
    print("  4. Prepare manuscript")
    print("="*80)


if __name__ == '__main__':
    import torch
    import torchvision
    main()
