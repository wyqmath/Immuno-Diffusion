# Immuno-Diffusion: A Biologically Inspired Privacy-Preserving Text-to-Image Framework (Proof-of-Concept)

This repository contains the proof-of-concept code for the "Immuno-Diffusion" project. It aims to apply mechanisms from the biological immune system (such as antigen recognition, antibody generation, and immune memory) to text-to-image diffusion models to enhance their privacy-preserving capabilities.

## Core Idea

Inspired by the three-stage process of the adaptive immune system, we aim to build a dynamic and intelligent privacy protection layer embedded within the text-to-image generation pipeline. The goal is to protect user privacy while maintaining high-quality image generation.

## Main Modules (`Immuno_Diffusion.py`)

The `Immuno_Diffusion.py` file implements the framework for the following core modules:

1.  **`PrivacyDetectionUnit` (PDU) - Simplified Version**:
    *   Simulates Antigen-Presenting Cells (APCs) and T-cell activation.
    *   **Current Implementation**: Detects privacy risks based on a configurable list of sensitive keywords found in text prompts. Outputs a risk score and dummy feature embeddings.
    *   The original concept involved BioBERT and knowledge graph integration, which are currently not part of this simplified version.

2.  **`PrivacyEnhancementUnit` (SEU) - Simplified Version**:
    *   Simulates B-cell generation of antibodies (i.e., privacy-preserving perturbations).
    *   **Current Implementation**: Based on the risk score from the PDU, injects a configurable level of Gaussian noise into the latent space of the diffusion model during the generation process. It does not currently use the memory signal.

3.  **`ImmuneMemoryModule`**:
    *   Simulates the memory function of the immune system.
    *   **Current Implementation**: Uses NeRF-inspired Fourier features to encode input features (intended to be high-risk patterns from PDU) and stores them in a memory bank.
    *   Allows querying for similar known patterns based on cosine similarity and can update the memory bank by merging or adding new patterns. The `validation_script.py` does not yet actively use this module to influence SEU behavior.

4.  **`ImmunoDiffusionModel` (Conceptual Wrapper)**:
    *   A conceptual `nn.Module` wrapper that initializes the base diffusion model components (tokenizer, text_encoder, VAE, UNet, scheduler from Hugging Face `diffusers`) along with the PDU, SEU, and ImmuneMemoryModule.
    *   Its `forward` method outlines a potential pipeline for integrating these immune-inspired modules into the diffusion generation process, including prompt encoding, PDU risk assessment, memory update/query, apoptosis check (placeholder), and the denoising loop with SEU intervention.
    *   This class is more of a structural guide and is not directly used for end-to-end image generation in `validation_script.py`; instead, `validation_script.py` implements its own generation loop incorporating the PDU and SEU.

## Advanced Concepts (Partially Implemented or Planned)

The codebase also includes placeholders or plans for more advanced biologically inspired mechanisms:

*   **`ApoptosisMechanism`**: A placeholder class is defined in `Immuno_Diffusion.py`. The conceptual `ImmunoDiffusionModel.forward()` includes a check for apoptosis, but the actual mechanism (e.g., disabling model parts) is not implemented.
*   **Epigenetic Regulation**: Placeholder function `epigenetic_prompt_encoding` exists. This concept involves exploring prompt encoding methods analogous to DNA encryption to enhance input robustness.
*   **Quorum Quenching**: Considered for distributed environments, not implemented in the current single-model codebase.
*   **Differentiable Privacy Engine**: Planned for more advanced noise injection methods (e.g., based on Langevin dynamics) to replace the current simple noise in SEU.

## Validation (`validation_script.py`)

A validation script (`validation_script.py`) is provided to demonstrate and evaluate the simplified framework. Its key functions include:

*   Loading a base diffusion model (e.g., "runwayml/stable-diffusion-v1-5") and its components (tokenizer, text encoder, VAE, UNet, scheduler).
*   Initializing and using the simplified `PrivacyDetectionUnit` (PDU) and `PrivacyEnhancementUnit` (SEU) from `Immuno_Diffusion.py`.
*   Loading prompts:
    *   From an external dataset file (`.csv` or `.txt`).
    *   Using a set of internal, hardcoded test prompts (default behavior when run directly or with `dataset_path="internal_test_prompts"`).
*   Generating images based on the prompts, with an option to enable/disable the PDU/SEU privacy mechanisms via the `--enable_privacy` flag.
*   Calculating evaluation metrics (currently as placeholders):
    *   **Fréchet Inception Distance (FID)**: The script includes a function `calculate_fid` which currently returns a placeholder value.
    *   **Semantic Leakage Index (SLI)**: The script includes a function `calculate_sli` which currently returns a placeholder value.
*   Saving the results (configuration arguments, FID/SLI scores) as a JSON file and a subset of generated sample images to a specified output directory.
*   Accepts various command-line arguments for configuration, including:
    *   `--dataset_path`: Path to the dataset or "internal_test_prompts".
    *   `--model_id`: Hugging Face model ID for the base diffusion model.
    *   `--sensitive_keywords_path`: Path to a file with sensitive keywords for SLI and optionally for PDU.
    *   `--pdu_sensitive_keywords`: A list of keywords for the PDU.
    *   `--seu_noise_level`: Noise level for the SEU.
    *   `--output_dir`: Directory to save results.
    *   `--enable_privacy`: Flag to enable PDU/SEU.
    *   `--calculate_fid`, `--calculate_sli`: Flags to trigger (placeholder) metric calculations.
    *   Other generation parameters like `batch_size`, `num_samples`, `image_size`, `guidance_scale`, `num_inference_steps`, `seed`.

The `main()` function in `validation_script.py` is set up to run with hardcoded basic settings (internal prompts, privacy disabled by default for this specific direct run) for quick testing.

## Goal

The ultimate goal of this project is to develop a text-to-image framework that can intelligently adapt to privacy threats and strike a good balance between image generation quality and privacy protection.

## Current Status

*   This codebase is a proof-of-concept.
*   **`Immuno_Diffusion.py`**:
    *   Contains simplified, functional versions of `PrivacyDetectionUnit` (keyword-based) and `PrivacyEnhancementUnit` (simple noise injection).
    *   Includes a functional `ImmuneMemoryModule` (NeRF-inspired Fourier features for encoding/querying patterns).
    *   `ApoptosisMechanism` is a placeholder class.
    *   `ImmunoDiffusionModel` is a conceptual wrapper demonstrating how modules *could* integrate, not a fully operational end-to-end model itself.
*   **`validation_script.py`**:
    *   Successfully loads a base diffusion model and the simplified PDU/SEU.
    *   Can generate images with or without these privacy modules enabled, based on user prompts or hardcoded examples.
    *   Saves generated images and (placeholder) metrics.
    *   The `ImmuneMemoryModule` is initialized if privacy is enabled but its output is not currently used to adapt SEU behavior within this script.
    *   FID and SLI calculations are placeholders.
*   The project demonstrates the basic integration of simplified immune-inspired privacy modules into a text-to-image generation pipeline for validation purposes. Further development is needed to implement more sophisticated versions of these modules and fully realize the advanced concepts. 