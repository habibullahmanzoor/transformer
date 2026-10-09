"""
Script to generate publication-ready figures from v4 corrected results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Create output directory
os.makedirs('/workspace/github__habibullahmanzoor__transformer/results/v4_corrected/figures', exist_ok=True)

def load_results():
    """Load all result files."""
    results_dir = '/workspace/uaap/results/v4_corrected'
    all_results = []
    
    for filename in os.listdir(results_dir):
        if filename.startswith('results_') and filename.endswith('.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                data = json.load(f)
                all_results.append(data)
    
    return all_results

def create_comparison_figure(results):
    """Create comparison figure between UAAP-GEN and Input-UAP."""
    seeds = []
    uaap_fr = []
    uaap_asr = []
    input_fr = []
    input_asr = []
    
    for r in results:
        seeds.append(r['seed'])
        uaap_fr.append(r['uaap_gen']['fr'])
        uaap_asr.append(r['uaap_gen']['asr'])
        input_fr.append(r['input_uap']['fr'])
        input_asr.append(r['input_uap']['asr'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Fooling Rate comparison
    x = np.arange(len(seeds))
    width = 0.35
    
    axes[0].bar(x - width/2, uaap_fr, width, label='UAAP-GEN', color='#1f77b4', alpha=0.8)
    axes[0].bar(x + width/2, input_fr, width, label='Input-UAP', color='#ff7f0e', alpha=0.8)
    axes[0].set_xlabel('Random Seed')
    axes[0].set_ylabel('Fooling Rate (FR)')
    axes[0].set_title('Fooling Rate Comparison\n(MNIST, 2-Layer ViT)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Seed {s}' for s in seeds])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ASR comparison
    axes[1].bar(x - width/2, uaap_asr, width, label='UAAP-GEN', color='#1f77b4', alpha=0.8)
    axes[1].bar(x + width/2, input_asr, width, label='Input-UAP', color='#ff7f0e', alpha=0.8)
    axes[1].set_xlabel('Random Seed')
    axes[1].set_ylabel('Attack Success Rate (ASR)')
    axes[1].set_title('Attack Success Rate Comparison\n(MNIST, 2-Layer ViT)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'Seed {s}' for s in seeds])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    fig.savefig('/workspace/github__habibullahmanzoor__transformer/results/v4_corrected/figures/comparison_fr_asr.png', dpi=300, bbox_inches='tight')
    fig.savefig('/workspace/github__habibullahmanzoor__transformer/results/v4_corrected/figures/comparison_fr_asr.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✅ Saved comparison figure")

def create_metrics_table(results):
    """Create a table of all metrics."""
    data = []
    for r in results:
        data.append({
            'Seed': r['seed'],
            'Method': 'UAAP-GEN',
            'Clean Acc': r['uaap_gen']['clean_acc'],
            'Adv Acc': r['uaap_gen']['adv_acc'],
            'FR': r['uaap_gen']['fr'],
            'ASR': r['uaap_gen']['asr']
        })
        data.append({
            'Seed': r['seed'],
            'Method': 'Input-UAP',
            'Clean Acc': r['input_uap']['clean_acc'],
            'Adv Acc': r['input_uap']['adv_acc'],
            'FR': r['input_uap']['fr'],
            'ASR': r['input_uap']['asr']
        })
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = df.values
    columns = df.columns
    
    # Format numeric columns
    for i, col in enumerate(columns):
        if col not in ['Seed', 'Method']:
            table_data[:, i] = [f"{v:.4f}" for v in table_data[:, i]]
    
    ax.table(cellText=table_data, colLabels=columns, 
             cellLoc='center', loc='center', colColours=["#f2f2f2"]*len(columns))
    ax.set_title('Detailed Metrics: UAAP-GEN vs Input-UAP\n(MNIST, 2-Layer ViT, 3 Seeds)', 
                 pad=20, fontsize=14)
    
    plt.tight_layout()
    fig.savefig('/workspace/github__habibullahmanzoor__transformer/results/v4_corrected/figures/metrics_table.png', dpi=300, bbox_inches='tight')
    fig.savefig('/workspace/github__habibullahmanzoor__transformer/results/v4_corrected/figures/metrics_table.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✅ Saved metrics table")

def create_clean_vs_adv_figure(results):
    """Create clean vs adversarial accuracy figure."""
    seeds = []
    uaap_clean = []
    uaap_adv = []
    input_clean = []
    input_adv = []
    
    for r in results:
        seeds.append(r['seed'])
        uaap_clean.append(r['uaap_gen']['clean_acc'])
        uaap_adv.append(r['uaap_gen']['adv_acc'])
        input_clean.append(r['input_uap']['clean_acc'])
        input_adv.append(r['input_uap']['adv_acc'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(seeds))
    width = 0.35
    
    ax.bar(x - width/2, uaap_clean, width, label='UAAP-GEN Clean', color='#1f77b4', alpha=0.8)
    ax.bar(x - width/2, [c - a for c, a in zip(uaap_clean, uaap_adv)], width, 
           bottom=uaap_adv, label='UAAP-GEN Adv', color='#1f77b4', alpha=0.4, hatch='//')
    
    ax.bar(x + width/2, input_clean, width, label='Input-UAP Clean', color='#ff7f0e', alpha=0.8)
    ax.bar(x + width/2, [c - a for c, a in zip(input_clean, input_adv)], width,
           bottom=input_adv, label='Input-UAP Adv', color='#ff7f0e', alpha=0.4, hatch='//')
    
    ax.set_xlabel('Random Seed')
    ax.set_ylabel('Accuracy')
    ax.set_title('Clean vs Adversarial Accuracy\n(MNIST, 2-Layer ViT)', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Seed {s}' for s in seeds])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('/workspace/github__habibullahmanzoor__transformer/results/v4_corrected/figures/clean_vs_adv.png', dpi=300, bbox_inches='tight')
    fig.savefig('/workspace/github__habibullahmanzoor__transformer/results/v4_corrected/figures/clean_vs_adv.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✅ Saved clean vs adv figure")

def main():
    print("Generating figures from v4 corrected results...")
    
    results = load_results()
    print(f"Loaded {len(results)} result files")
    
    # Create all figures
    create_comparison_figure(results)
    create_metrics_table(results)
    create_clean_vs_adv_figure(results)
    
    print("\n✅ All figures generated!")
    print("\nFigures saved to:")
    print("  - results/v4_corrected/figures/comparison_fr_asr.png")
    print("  - results/v4_corrected/figures/comparison_fr_asr.pdf")
    print("  - results/v4_corrected/figures/metrics_table.png")
    print("  - results/v4_corrected/figures/metrics_table.pdf")
    print("  - results/v4_corrected/figures/clean_vs_adv.png")
    print("  - results/v4_corrected/figures/clean_vs_adv.pdf")

if __name__ == '__main__':
    main()
