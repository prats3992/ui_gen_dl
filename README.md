# Akin-Enhanced: UI Wireframe Generation with Multi-Head Attention and Diffusion Models

This project reproduces the results of *Akin: Generating UI Wireframes from UI Design Patterns Using Deep Learning* by Gajjar et al., with enhancements to improve UI wireframe generation. The original work uses a Self-Attention GAN (SAGAN) trained on UI wireframes for specific design patterns. Our goal is to replace self-attention with multi-head attention and experiment with diffusion models as an alternative to GANs for more refined UI generation.

## Table of Contents
- [Overview](#overview)
- [Original Model](#original-model)
- [Enhancements](#enhancements)
  - [Multi-Head Attention](#multi-head-attention)
  - [Diffusion Model](#diffusion-model)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)

## Overview
This repository provides an enhanced approach to the UI wireframe generation model presented in the original Akin project. The main improvements involve integrating a multi-head attention mechanism and experimenting with diffusion models to generate high-quality wireframes for UI design patterns. This work aligns with the baseline set by the original paper.

## Original Model
The original Akin model utilizes a Self-Attention Generative Adversarial Network (SAGAN) to produce UI wireframes from semantic representations. It is trained on 500 UI wireframes for five Android UI design patterns (Splash Screen, Login, Account Registration, Product Catalog, Product Page) from the RICO dataset. Key evaluation metrics include the Inception Score (IS) and Fréchet Inception Distance (FID).

## Enhancements

### Multi-Head Attention
To enhance the model's representation and generation capabilities, we replace the self-attention mechanism in SAGAN with a multi-head attention mechanism. Multi-head attention allows the model to focus on different aspects of the input simultaneously, potentially improving the quality and diversity of generated wireframes.

### Diffusion Model
As an alternative to GANs, we explore diffusion models for UI wireframe generation. Diffusion models have shown promise in generating high-quality images and may offer advantages in terms of generation stability and diversity. This implementation involves adapting a diffusion model to the wireframe generation task, comparing its performance with the original SAGAN-based model.

## Setup
1. Clone the repository:

2. Install the required packages:

3. Go through the complete codebase and understand the implementation of the original SAGAN model, the multi-head attention model, and the diffusion model.

## Usage
1. Reproduce Original Results: Use the script train_sagan.py to train a SAGAN model with self-attention, as done in the original Akin project.

```bash
python train_sagan.py --config configs/original_sagan.yaml
```
2. Train with Multi-Head Attention: Train a modified model that uses multi-head attention instead of self-attention.

```bash
python train_multihead_attention.py --config configs/multihead_attention.yaml
```

3. Experiment with Diffusion Model: Run the diffusion model training script to generate wireframes with a diffusion approach.

```bash
python train_diffusion.py --config configs/diffusion_model.yaml
```
## Results
Results will be evaluated using Inception Score (IS) and Fréchet Inception Distance (FID) to assess the quality and similarity of generated wireframes compared to designer-made samples.
