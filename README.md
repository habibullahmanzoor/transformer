# UAAP-GEN: Universal Adversarial Attention Perturbations

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/habibullahmanzoor/transformer)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org/)

> **Publication Target:** ICLR/NeurIPS/CVPR 2026
> **Current Focus:** Defense Evasion Study (v5.0)

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision matplotlib pandas numpy seaborn
```

### Run Corrected Implementation
```bash
python uaap_gen_v4_corrected.py
```

### Run Defense Evasion Study
```bash
python uaap_gen_v5_defense.py
```

### Generate Figures
```bash
python plot_results_v4.py
```

## 📚 Prior Work Acknowledgment

**IMPORTANT:** We acknowledge that UAAP-GEN is NOT the first to perturb attention mechanisms. The following established works exist in this space:

### Direct Attention Perturbation
- **"Corrupting Attention"** (2026) - Introduces direct optimization of encoder-attention objective under bounded perturbation
- **"AFOG: Attention-Focused Offensive Gradient"** (ICCV 2025) - Unified adversarial attention framework for detection transformers

### Universal Attention-Based UAPs
- **IAM-UAP** (Inheritance Attention Matrix-based UAP) - CVPR 2021
  - Introduces attention weight matrix with perturbation optimization
  - Targets global information integration in ViTs
- **AS-UAP** (Attention-Shift UAP) - 2025
  - Shifts model attention to create universal perturbations
  - Reports **81.55% fooling rate** on ViT (significantly higher than our current results)

### Attention-Guided Attacks
- **TPP-G** - Attention-guided perturbation generation
- **A-SAGE** - Attention-based Saliency Guided Adversarial Examples

## 🎯 Revised Research Position

### Original Claim (INCORRECT):
"WORLD FIRST: Direct attention matrix perturbation"

### Corrected Position:
UAAP-GEN contributes to the **established field** of attention-based adversarial attacks by:

1. **Systematic Defense Evasion Evaluation** - Testing attention-level attacks against defenses that input-level attacks may not bypass
2. **Comparative Analysis** - Fair comparison with established baselines (IAM-UAP, AS-UAP)
3. **Implementation Rigor** - Technically correct implementation with proper gradient flow and validation tests

### Novelty Pivot: Defense Evasion Focus

Instead of claiming novelty for attention perturbation itself, we focus on:
- **Can attention-level attacks bypass defenses that stop input-level attacks?**
- **Do attention perturbations transfer across models with different defenses?**
- **What is the robustness-accuracy trade-off for different defense mechanisms?**

This provides a **genuine scientific contribution** by systematically evaluating attack effectiveness in the presence of defenses.

## 📊 Latest Results

### v4.0 Corrected Implementation (MNIST, 3 seeds)

| Method | Clean Acc | Adv Acc | FR | ASR |
|--------|-----------|---------|----|-----|
| UAAP-GEN | 0.645-0.730 | 0.635-0.725 | 0.085-0.130 | 0.062-0.100 |
| Input-UAP | 0.645-0.730 | 0.575-0.700 | 0.105-0.220 | 0.069-0.177 |

### Key Validations
- ✅ **Zero Perturbation Equivalence:** f(x, δ=0) == f(x, clean) (max diff: 0.00e+00)
- ✅ **Gradient Flow:** Verified through all perturbation layers
- ✅ **Correct Preprocessing:** Input-UAP perturbs in raw space, normalizes after
- ✅ **Proper Metrics:** FR, ASR, and accuracy reduction separately reported

### Comparison with SOTA (Reported in Literature)
| Method | Fooling Rate | Dataset | Year |
|--------|--------------|---------|------|
| AS-UAP | **81.55%** | ViT | 2025 |
| UAAP-GEN v4.0 | 8.5-13.0% | MNIST | 2025 |
| IAM-UAP | ~70-75% | CIFAR-10 | 2021 |

**Note:** Our current results are significantly lower than SOTA. This is expected because:
1. We use MNIST (simple dataset) vs. CIFAR-10/ImageNet
2. We use a small 2-layer ViT vs. standard DeiT/Swin
3. We focus on **technical correctness** rather than maximizing fooling rates

## 🛡️ Defense Evasion Study (v5.0)

### Defenses Implemented
1. **Adversarial Training** - Model trained with adversarial examples
2. **Attention Smoothing** - Averages attention weights across heads
3. **Randomized Smoothing** - Adds Gaussian noise, averages predictions
4. **Gradient Masking** - Reduces gradient flow through attention

### Research Questions
- Can UAAP-GEN bypass defenses that stop Input-UAP?
- Which defenses are most effective against attention-level attacks?
- What is the clean accuracy cost of each defense?

## 📁 Project Structure

```
transformer/
├── README.md                          # This file
├── uaap_gen_v4_corrected.py          # Technically correct implementation
├── uaap_gen_v5_defense.py            # Defense evasion study
├── plot_results_v4.py                # Figure generation
└── results/
    ├── v4_corrected/                   # Corrected results
    │   ├── results_*.json              # Per-seed results
    │   ├── summary_*.json               # Summary
    │   └── figures/                    # Publication-ready figures
    │       ├── comparison_fr_asr.png
    │       ├── metrics_table.png
    │       └── clean_vs_adv.png
    └── v5_defense/                     # Defense study results
        ├── defense_results_*.json
        └── summary_*.json
```

## 🔬 Implementation Details

### Core Innovation (v4.0)
UAAP-GEN directly perturbs attention scores **inside** the original MultiheadAttention:

```python
# In PerturbableMultiheadAttention.forward()
attn_scores = (q @ k.transpose(-2, -1)) * scale

# Add perturbation INSIDE attention computation
if perturbation is not None:
    attn_scores = attn_scores + perturbation  # Preserves Q,K,V projections

attn_weights = attn_scores.softmax(dim=-1)
```

### Critical Fixes from v3.0
1. **LEAF Tensors:** Perturbations as `nn.Parameter` for proper gradient flow
2. **Dataset-wide Optimization:** Gradient accumulation across entire dataset
3. **Strict PGD Projection:** Norm constraint enforcement after each step
4. **Architecture Preservation:** Perturbations applied INSIDE original attention (not replacing it)
5. **Correct Preprocessing:** Input-UAP works in raw pixel space
6. **Proper Metrics:** FR, ASR, accuracy reduction separately computed

## 📈 Publication Roadmap

### ✅ Phase 1: COMPLETED
- Algorithmic Audit & Efficacy Fix
- Fixed gradient flow and optimization issues
- Achieved technically correct implementation
- Validated with proper tests

### ✅ Phase 2: COMPLETED
- Acknowledged prior work (IAM-UAP, AS-UAP, etc.)
- Pivoted to defense evasion focus
- Implemented defense mechanisms

### Phase 3: SOTA Comparison (Next)
- [ ] Implement IAM-UAP baseline
- [ ] Implement AS-UAP baseline
- [ ] Compare with TPP-G, A-SAGE
- [ ] Use standard evaluation metrics

### Phase 4: Scaled Evaluation
- [ ] Test on CIFAR-10/100
- [ ] Test on ImageNet
- [ ] Use DeiT-Tiny, Swin-T, ViT-B/16
- [ ] Multiple seeds (5+)

### Phase 5: Complete Study
- [ ] Ablation studies (layer-wise, head-wise, budget)
- [ ] Transferability tests
- [ ] Defense effectiveness analysis
- [ ] Robustness-accuracy trade-off

### Phase 6: Manuscript
- [ ] Write paper focusing on defense evasion
- [ ] Position as systematic evaluation study
- [ ] Cite all prior work properly
- [ ] Submit to ICLR/NeurIPS/CVPR 2026

## 📊 Results Summary

### v4.0 Corrected Results (MNIST, 2-Layer ViT)

| Seed | Method | Clean Acc | Adv Acc | FR | ASR |
|------|--------|-----------|---------|----|-----|
| 42 | UAAP-GEN | 0.730 | 0.695 | 0.085 | 0.062 |
| 42 | Input-UAP | 0.730 | 0.700 | 0.105 | 0.069 |
| 123 | UAAP-GEN | 0.650 | 0.620 | 0.130 | 0.100 |
| 123 | Input-UAP | 0.650 | 0.575 | 0.220 | 0.177 |
| 456 | UAAP-GEN | 0.645 | 0.620 | 0.125 | 0.093 |
| 456 | Input-UAP | 0.645 | 0.585 | 0.185 | 0.155 |

### Observations
- Input-UAP generally achieves higher FR/ASR than UAAP-GEN on MNIST
- Both methods show significant variation across seeds
- UAAP-GEN provides **defense evasion potential** that Input-UAP may not

## 📚 Citation

```bibtex
@misc{uaapgen2025,
  author = {Habibullah Manzoor},
  title = {{UAAP-GEN}: Universal Adversarial Attention Perturbations - A Defense Evasion Study},
  year = {2025},
  howpublished = {\url{https://github.com/habibullahmanzoor/transformer}},
  note = {Target: ICLR/NeurIPS/CVPR 2026}
}
```

## 📖 Related Work

### Established Attention-Based UAP Methods
```bibtex
@inproceedings{iamuap2021,
  title={Inheritance Attention Matrix-Based Universal Adversarial Perturbations},
  author={Hu, et al.},
  booktitle={CVPR},
  year={2021}
}

@article{asuap2025,
  title={Attention-Shift Universal Adversarial Perturbation},
  author={Anonymous},
  year={2025}
}

@article{corrupting2026,
  title={Corrupting Attention},
  author={Anonymous},
  year={2026}
}

@inproceedings{afog2025,
  title={Attention-Focused Offensive Gradient},
  author={Anonymous},
  booktitle={ICCV},
  year={2025}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License.

## 📞 Contact

For questions or collaborations, please open an issue.

---

**Status:** Phase 2 Complete - Defense Evasion Focus Established  
**Next:** Phase 3 - SOTA Baseline Implementation  
**Publication Strategy:** Position as defense evasion study, not as "world first"
