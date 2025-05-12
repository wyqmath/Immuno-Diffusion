# Immuno-Diffusion: A Biologically Inspired Privacy-Preserving Text-to-Image Framework (Proof-of-Concept)

This repository contains the proof-of-concept code for the "Immuno-Diffusion" project. It aims to apply mechanisms from the biological immune system (such as antigen recognition, antibody generation, and immune memory) to text-to-image diffusion models to enhance their privacy-preserving capabilities.

## Core Idea

Inspired by the three-stage process of the adaptive immune system, we aim to build a dynamic and intelligent privacy protection layer embedded within the text-to-image generation pipeline. The goal is to protect user privacy while maintaining high-quality image generation.

## Main Modules (`Immuno_Diffusion.py`)

The current code implements the initial framework for the following core modules:

1.  **`PrivacyDetectionUnit` (PDU)**:
    *   Simulates Antigen-Presenting Cells (APCs) and T-cell activation.
    *   Utilizes BioBERT for text input processing and optionally integrates a knowledge graph (ConceptNet GAT) to identify potential privacy risks, outputting a risk score.

2.  **`PrivacyEnhancementUnit` (SEU)**:
    *   Simulates B-cell generation of antibodies (i.e., privacy-preserving perturbations).
    *   Based on the risk score from the PDU, dynamically injects noise or other perturbations into the latent space of the diffusion model to interfere with the generation of sensitive information.

3.  **`ImmuneMemoryModule`**:
    *   Simulates the memory function of the immune system.
    *   Uses NeRF-inspired Fourier features to encode and store detected high-risk patterns.
    *   Allows querying for similar known threats, providing more targeted defense guidance to the SEU.

## Advanced Concepts (Planned)

The codebase also includes placeholders or plans for more advanced biologically inspired mechanisms:

*   **Apoptosis**: Triggering self-destruction or disabling parts of the model upon detecting persistent high risks or adversarial attacks.
*   **Epigenetic Regulation**: Exploring prompt encoding methods analogous to DNA encryption to enhance input robustness.
*   **Quorum Quenching**: Considering the application of defense strategies in distributed environments.
*   **Differentiable Privacy Engine**: Planning to use more advanced noise injection methods based on Langevin dynamics.

## Validation (`validation_script.py`)

A validation script (`validation_script.py`) is provided to evaluate the framework's performance. Its key functions include:

*   Loading the base diffusion model components and the Immuno-Diffusion modules (PDU, SEU, Memory).
*   Loading a dataset of prompts (and optionally sensitive flags/keywords).
*   Generating images based on the prompts, with or without the privacy mechanisms enabled.
*   Calculating evaluation metrics:
    *   **Fréchet Inception Distance (FID)**: Measures image quality and realism by comparing generated images to a reference dataset. (Currently uses placeholder values).
    *   **Semantic Leakage Index (SLI)**: Aims to quantify how much sensitive semantic information from the prompt is leaked into the generated image, often using CLIP similarity between sensitive prompts and their outputs. (Currently uses placeholder values).
*   Saving the results (metrics, configuration) and sample generated images.

The script accepts command-line arguments for model paths, dataset paths, output directory, batch size, number of samples, and whether to enable the privacy features.

## Goal

The ultimate goal of this project is to develop a text-to-image framework that can intelligently adapt to privacy threats and strike a good balance between image generation quality and privacy protection.

## Current Status

This codebase is currently in the proof-of-concept and module development stage. The Immuno-Diffusion components are defined but not yet fully integrated into a complete, end-to-end diffusion model pipeline. The validation script provides a structure for evaluation but currently uses placeholder logic for model loading, generation, and metric calculation. 