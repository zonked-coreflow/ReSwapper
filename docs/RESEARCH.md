# ReSwapper v2 Research Notes

## Papers Surveyed (2024-2026)

### Face Swapping Architecture
| Paper | Year | Type | Resolution | FID | ID Retrieval | Inference | arxiv |
|-------|------|------|-----------|-----|-------------|-----------|-------|
| DreamID | 2025 | Diffusion (1-step) | 512 | 4.69 | 99.9% | 0.6s | 2504.14509 |
| ID-Constrained | 2026 | Diffusion | 512 | 3.61 | 97.9% | Multi-step | 2503.22179 |
| REFace | 2024 | Diffusion | 512 | 5.53 | 95.4% | 4.7s | 2409.07269 |
| SelfSwapper | 2024 | GAN (ADM) | 256 | 21.22 | - | Fast | 2402.07370 |
| DynamicFace | 2025 | Diffusion | 512 | - | 99.2% | Multi-step | 2501.08553 |
| AlphaFace | 2026 | GAN | 256 | 2.71 | 98.8% | 24ms | 2601.16429 |
| VividFace | 2024 | Diffusion | 512 | - | 78.3% | Multi-step | 2412.11279 |
| CanonSwap | 2025 | GAN | 512 | - | - | 71ms | 2507.02691 |
| DiffSwap++ | 2025 | Diffusion | 512 | 6.42 | 94.7% | Multi-step | 2511.05575 |
| E4S | 2024 | GAN (StyleGAN) | 1024 | - | - | Fast | 2310.15081 |

### GAN Architecture & Training
| Paper | Year | Key Contribution | arxiv |
|-------|------|-----------------|-------|
| R3GAN | 2025 | Modern GAN: RpGAN + R1+R2, no normalization, fix-up init | 2501.05441 |
| HP-GAN | 2026 | Frozen DINOv2/ViT discriminator, FID 1.69 on FFHQ | 2602.03039 |
| LadaGAN | 2024 | Linear additive-attention for O(N) GANs | 2401.09596 |
| GAT | 2025 | ViT-based GAN, FID 2.96 ImageNet-256 | 2509.24935 |
| FCDM | 2026 | ConvNeXt U-Net, 50% fewer FLOPs than DiT | 2603.09408 |

### Identity Encoders
| Model | Year | Type | IJB-C TAR@1e-4 | Drop-in | Source |
|-------|------|------|----------------|---------|--------|
| BlendFace | 2023 | iResNet-50 (swap-specific) | N/A | HIGH | ICCV 2023 |
| TopoFR | 2024 | iResNet-100 | 97.60% | HIGH | NeurIPS 2024 |
| LVFace-B | 2025 | ViT-B | 97.70% | MEDIUM | ICCV 2025 |
| AdaFace | 2022 | iResNet-100 | 97.66% | HIGH | CVPR 2022 |
| TransFace-L | 2023 | ViT-L | 97.61% | MEDIUM | ICCV 2023 |

Key finding: BlendFace is specifically designed for face swapping — fixes ArcFace's identity-attribute entanglement (positive pair 0.82 vs swapped pair 0.80, small gap = clean identity).

### Perceptual Losses
| Loss | Source | Improvement | Notes |
|------|--------|-------------|-------|
| P-DINO | PixelGen 2026 | FID 10.0→7.46 | DINOv2-B layer 12 patch cosine distance |
| DreamSim | 2023 | 96.2% vs LPIPS 70.7% human agreement | Ensemble DINO+CLIP+OpenCLIP with LoRA |
| R-CLIP_F | 2025 | Adversarially robust | 90.6% NIGHTS zero-shot |

### Face Parsing Models
| Model | F1 Score | Speed | Params |
|-------|----------|-------|--------|
| FaceXFormer | 92.01 | 33 FPS | 109M |
| SegFace (Swin-T) | 88.96 | - | ~30M |
| SegFace-Mobile | 87.91 | 96 FPS | Light |
| BiSeNet (legacy) | ~83 | Fast | ~13M |

### Occlusion Handling
- **FaceMat** (2025, 2508.03055): Alpha matting for soft occlusion boundaries
- **SelfSwapper** (2024, 2402.07370): Explicit M = M_ras - M_occ, perforation confusion
- **VividFace** (2024): Comprehensive occlusion augmentation with temporal patterns
- **CelebAMat**: Synthetic occlusion dataset from SIMD, AM2k, HIU hand seg, DTD textures

### VAE Comparison
| VAE | Year | Channels | Spatial | PSNR (general) | PSNR (FFHQ) | Params | License |
|-----|------|----------|---------|----------------|-------------|--------|---------|
| SD 1.5 ft-MSE | 2022 | 4 | 8x | ~24.5 | ~24-25 | ~84M | Apache 2.0 |
| SDXL | 2023 | 4 | 8x | ~26.8 | ~26-27 | ~84M | Open |
| FLUX.1 | 2024 | 16 | 8x | ~36.1 | ~35+ | ~160M | Apache 2.0 |
| FLUX.2 | 2025 | 32 | 8x/16x | ~38.3 | ~37+ | ~160M+ | FLUX.2 license |
| Cosmos CI8x8 | 2025 | 16 | 8x | ~33.1 | **39.67** | 77M | NVIDIA open |
| DC-AE f32c32 | 2024 | 32 | 32x | ~23.9 | ~31 | varies | MIT |

Z-Image-Turbo uses FLUX.1 VAE unmodified (confirmed in arXiv:2511.22699 Section 4.1). Cosmos CI8x8 has best face reconstruction (PSNR 39.67 FFHQ, 46.47 CelebA-HQ) but is an AE not VAE.

### Geometric/Attribute Losses
- **Gaze preservation** (2402.03188): Pretrained gaze estimator, cosine distance on gaze vectors
- **Expression preservation**: 3DDFA-V3 expression coefficient L2 (DynamicFace)
- **68-point landmark L2**: dlib/mediapipe (DiffSwap++)
- **CLIP contrastive**: VLM-generated descriptions (AlphaFace)

## Key Design Decisions

1. **Latent space (FLUX VAE)**: 16ch, 8x spatial, same space as Z-Image. 4x more info than SD 1.5.
2. **R3GAN blocks**: No normalization, fix-up init, grouped conv. Simpler AND better.
3. **Projected GAN (DINOv2)**: Frozen backbone + trainable heads. HP-GAN FID 1.69.
4. **BlendFace identity**: Designed for face swap, fixes attribute entanglement.
5. **Self-supervised training**: SelfSwapper + DreamID approaches, no InSwapper dependency.
6. **Occlusion**: FaceMat alpha matting + synthetic augmentation + learned swap mask.
