# Immuno-Diffusion: A Biologically Inspired Privacy-Preserving Text-to-Image Framework (Proof-of-Concept)

This repository contains the proof-of-concept code for the "Immuno-Diffusion" project. It aims to apply mechanisms from the biological immune system (such as antigen recognition, antibody generation, and immune memory) to text-to-image diffusion models to enhance their privacy-preserving capabilities.

## Core Idea

Inspired by the three-stage process of the adaptive immune system, we aim to build a dynamic and intelligent privacy protection layer embedded within the text-to-image generation pipeline. The goal is to protect user privacy while maintaining high-quality image generation.

## Main Modules (`Immuno_Diffusion.py`)

The `Immuno_Diffusion.py` file implements the framework for the following core modules:

1.  **`PrivacyDetectionUnit` (PDU)**:
    *   Simulates Antigen-Presenting Cells (APCs) and T-cell activation.
    *   **Current Implementation**: This module is designed as a trainable classifier. It uses a pre-trained BioBERT model to encode text prompts and a Graph Attention Network (GAT) to encode conceptual graph information. These features are then combined to produce a privacy risk score.
    *   **Inconsistency Note**: There is currently a discrepancy between the module's design and its usage. The `__init__` method accepts a list of sensitive keywords, but these are not used in the `forward` pass to determine the risk score. This logic needs to be clarified and aligned.

2.  **`PrivacyEnhancementUnit` (SEU) - Simplified Version**:
    *   Simulates B-cell generation of antibodies (i.e., privacy-preserving perturbations).
    *   **Current Implementation**: Based on the risk score from the PDU, injects a configurable level of Gaussian noise into the latent space of the diffusion model during the generation process. It does not currently use the memory signal.

3.  **`ImmuneMemoryModule`**:
    *   Simulates the memory function of the immune system.
    *   **Current Implementation**: Uses NeRF-inspired Fourier features to encode input features (intended to be high-risk patterns from PDU) and stores them in a memory bank.
    *   Allows querying for similar known patterns based on cosine similarity and can update the memory bank by merging or adding new patterns. The `validation_script.py` does not yet actively use this module.

4.  **`ImmunoDiffusionModel` (Conceptual Wrapper)**:
    *   A conceptual `nn.Module` wrapper that initializes the base diffusion model components (tokenizer, text_encoder, VAE, UNet, scheduler from Hugging Face `diffusers`) along with the PDU, SEU, and ImmuneMemoryModule.
    *   Its `forward` method outlines a potential pipeline for integrating these immune-inspired modules into the diffusion generation process.
    *   This class serves as a structural guide and is **not** directly used for end-to-end image generation in `validation_script.py`; instead, the validation script implements its own generation loop.

## Advanced Concepts (Partially Implemented or Planned)

The codebase also includes placeholders or plans for more advanced biologically inspired mechanisms:

*   **`ApoptosisMechanism`**: A placeholder class is defined. The conceptual `ImmunoDiffusionModel.forward()` includes a check for apoptosis, but the actual mechanism (e.g., disabling model parts) is not implemented.
*   **Epigenetic Regulation**: A placeholder function `epigenetic_prompt_encoding` exists, exploring robust prompt encoding.
*   **Quorum Quenching**: Considered for distributed environments, not implemented in the current single-model codebase.
*   **Differentiable Privacy Engine**: Planned for more advanced noise injection methods (e.g., based on Langevin dynamics) to replace the current simple noise in SEU.

## Validation (`validation_script.py`)

A validation script is provided to demonstrate and evaluate the framework. Its key functions include:

*   Loading a base diffusion model (e.g., "runwayml/stable-diffusion-v1-5").
*   Initializing and using the `PrivacyDetectionUnit` (PDU) and `PrivacyEnhancementUnit` (SEU).
*   Loading prompts from a file (`.csv`, `.txt`) or using internal hardcoded test prompts.
*   Generating images with an option to enable/disable the PDU/SEU privacy mechanisms via the `--enable_privacy` flag.
*   Calculating evaluation metrics:
    *   **Semantic Leakage Index (SLI)**: Implemented using a CLIP model to measure the similarity between generated images and sensitive keywords.
    *   **Attack Rejection Degree (ARD)**: Implemented to assess the model's ability to reject generating content for prompts deemed as attacks.
    *   **Fréchet Inception Distance (FID)**: The function `calculate_fid` is a **placeholder** and returns a fixed value.
*   Saving results (config, scores) as a JSON file and saving generated sample images.
*   Accepts various command-line arguments for configuration (e.g., `model_id`, `pdu_sensitive_keywords`, `seu_noise_level`, `output_dir`, `enable_privacy`).

The `main()` function in `validation_script.py` is set up for a quick test run using hardcoded settings.

## Current Status

*   This codebase is a proof-of-concept.
*   **`Immuno_Diffusion.py`**:
    *   Contains a trainable `PrivacyDetectionUnit` (using BioBERT and GATs), a functional `PrivacyEnhancementUnit` (simple noise injection), and a complete `ImmuneMemoryModule`.
    *   `ApoptosisMechanism` is a placeholder.
    *   `ImmunoDiffusionModel` is a conceptual wrapper, not used for validation runs.
*   **`validation_script.py`**:
    *   Successfully loads a base diffusion model and the PDU/SEU modules.
    *   Can generate images with or without these privacy modules enabled. (Note: A bug that prevented the SEU from being called was recently fixed).
    *   Saves generated images and evaluation results.
    *   **The `ImmuneMemoryModule` is not currently integrated into the validation script's generation loop.**
    *   SLI and ARD metrics are implemented, while the FID calculation is a placeholder.
*   The project demonstrates the basic integration of immune-inspired privacy modules into a text-to-image pipeline. Further development is needed to resolve inconsistencies (like the PDU keyword issue), implement more sophisticated versions of the modules (like using the memory module), and fully realize the advanced concepts. 