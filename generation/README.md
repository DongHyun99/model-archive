# Generation

## Models

Image 생성에서 자주 사용하는 모델은 크게 두가지 종류가 있습니다.

- ### Generative Adversarial Networks(GAN)  

    - GAN의 장점  
        - Diffusion model에 비해 메모리 효율이 높습니다. (더 적은 메모리로 품질이 좋은 이미지 생성이 가능.)
        - 이미지의 생성이 매우 빠릅니다. (train time/inference time 모두 압도적으로 빠름.)
        - 코드의 구조도 비교적 간단하고 loss function이나 구조도 직관적이여서 이해하기 쉽습니다.

- ### Diffusion models

    - Diffusion models의 장점
        - GAN에 비해 학습이 매우 안정적입니다. (GAN은 hyperparameter에 따라 quality 차이가 좀 많이 납니다.)
        - Improved DDPM 이후로 quality 뿐아니라 diversity도 높은 이미지 생성이 가능합니다. (즉 고품질이면서 다양성이 높은 이미지 생성이 가능.)  

## Papers

읽어볼만한 논문입니다.

### GAN

- [Generative Adversarial Networks (2014)](https://arxiv.org/abs/1406.2661): GAN  

- [Conditional Generative Adversarial Nets (2014)](https://arxiv.org/abs/1411.1784): CGAN  

- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (2015)](https://arxiv.org/abs/1511.06434): DCGAN  

- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498): Improved GAN

- [Image-to-Image Translation with Conditional Adversarial Networks (2016)](https://arxiv.org/abs/1611.07004): pix2pix (patchGAN)  

- [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (2016)](https://arxiv.org/abs/1609.04802): SRGAN

    - [Perceptual Losses for Real-Time Style Transfer and Super-Resolution (2016)](https://arxiv.org/abs/1603.08155): Perceptual Loss
    - [StyTr2: Image Style Transfer with Transformers (2021)](https://arxiv.org/abs/2105.14576): StyleTr2 (Transformer based Perceptual Loss)

- [Wasserstein GAN (2017)](https://arxiv.org/abs/1701.07875): WGAN  

- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks(2017)](https://arxiv.org/abs/1703.10593): CycleGAN  

- [Progressive Growing of GANs for Improved Quality, Stability, and Variation (2017)](https://arxiv.org/abs/1710.10196): PGGAN (proGAN)

- [Unsupervised Image-to-Image Translation Networks (2017)](https://arxiv.org/abs/1703.00848): UNIT

- [A Style-Based Generator Architecture for Generative Adversarial Networks (2018)](https://arxiv.org/abs/1812.04948): StyleGAN  

    - [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization (2017)](https://arxiv.org/abs/1703.06868): AdaIN  

- [Multimodal Unsupervised Image-to-Image Translation (2018)](https://arxiv.org/abs/1804.04732): MUNIT

- [Few-Shot Unsupervised Image-to-Image Translation (2019)](https://arxiv.org/abs/1905.01723): FUNIT  

- [Analyzing and Improving the Image Quality of StyleGAN (2019)](https://arxiv.org/abs/1912.04958): StyleGAN2  

- [MSG-GAN: Multi-Scale Gradients for Generative Adversarial Networks (2019)](https://arxiv.org/abs/1903.06048): MSG-GAN

- [Non-saturating GAN training as divergence minimization (2020)](https://arxiv.org/abs/2010.08029): Non-saturating GAN

- [Taming Transformers for High-Resolution Image Synthesis (2020)](https://arxiv.org/abs/2012.09841): VQ-GAN  

    - [Neural Discrete Representation Learning (2017)](https://arxiv.org/abs/1711.00937): VQ-VAE

- [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold (2023)](https://arxiv.org/abs/2305.10973): Drag GAN

### Diffusion Models

- [Understanding Diffusion Models: A Unified Perspective (2022)](https://arxiv.org/abs/2208.11970): VAE, HVAE, VDM, SDE, Classifier Guidance, Classifier-free Guidance 수식 정리  

- [Denoising Diffusion Probabilistic Models (2020)](https://arxiv.org/abs/2006.11239): DDPM

- [Denoising Diffusion Implicit Models (2020)](https://arxiv.org/abs/2010.02502): DDIM

- [Score-Based Generative Modeling through Stochastic Differential Equations (2020)](https://arxiv.org/abs/2011.13456): SDE

- [Variational Diffusion Models (2021)](https://arxiv.org/abs/2107.00630): VDM

- [UNIT-DDPM: UNpaired Image Translation with Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2104.05358): UNIT-DDPM

- [Improved Denoising Diffusion Probabilistic Models (2021)](https://arxiv.org/abs/2102.09672): Improved DDPM

- [Diffusion Models Beat GANs on Image Synthesis (2021)](https://arxiv.org/abs/2105.05233): Classifier Guidance (guided-diffusion)  

- [Score-based Generative Modeling in Latent Space (2021)](https://arxiv.org/abs/2106.05931): LSGM  

- [High-Resolution Image Synthesis with Latent Diffusion Models (2021)](https://arxiv.org/abs/2112.10752): LDM (Stable Diffusion)

- [Zero-Shot Text-to-Image Generation (2021)](https://arxiv.org/abs/2102.12092): DALL-E

    - [Learning Transferable Visual Models From Natural Language Supervision (2021)](https://arxiv.org/abs/2103.00020): CLIP

- [Classifier-Free Diffusion Guidance (2022)](https://arxiv.org/abs/2207.12598): Classifier-free Guidance

- [Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise(2022)](https://arxiv.org/abs/2208.09392): Cold Diffusion

- [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps (2022)](https://arxiv.org/abs/2206.00927): DPM-Solver  

    - [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models (2022)](https://arxiv.org/abs/2211.01095): DPM-Solver++

- [Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models (2022)](https://arxiv.org/abs/2211.17091): DG

- [Consistency Models (2023)](https://arxiv.org/abs/2303.01469): Consistency Models

- [StyleDiffusion: Controllable Disentangled Style Transfer via Diffusion Models (2023)](https://arxiv.org/abs/2308.07863): StyleDiffusion

- [Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion (2023)](https://arxiv.org/abs/2310.02279): CTM

## Hackathon  

해커톤에서 나올만한 Image Geneation Task와 관련 모델입니다.

### Image Inpainting

- [Palette](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)

- [CM-GAN](https://github.com/htzheng/CM-GAN-Inpainting)

- [LaMa](https://github.com/advimman/lama)

    -[LaMa Feature Refinement](https://github.com/advimman/lama/pull/112)

### Super Resolution

- [HAT](https://github.com/XPixelGroup/HAT)

- [SwinFIR](https://github.com/Zdafeng/SwinFIR)

### Denoising

- [Restormer](https://github.com/swz30/restormer)

### Colorization

- [Palette](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)


## Tips

### Using Transformer based Perceptual Loss

Perceptual Loss는 VGG19의 feauture에 대한 L2 Distance를 측정하는 Loss입니다.  
VGG19 대신 Transformer 기반 모델에서 사용할 경우 성능이 향상된다는 연구가 있습니다. (https://github.com/diyiiyiii/StyTR-2)  

### Using Mix Loss for Reconstruction Loss  

Super Resolution에서 사용한 Loss로 MS-SSIM과 L1 Loss를 함께 사용하면 성능이 향상된다는 연구가 있습니다. (https://github.com/psyrocloud/MS-SSIM_L1_LOSS)  