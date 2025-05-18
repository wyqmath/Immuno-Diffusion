#!/usr/bin/env python
# coding: utf-8

import torch
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import pathlib # For save_results to handle Path objects if any in args

from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
# from diffusers.utils import randn_tensor # -> Replaced with torch.randn
from transformers import CLIPTextModel, CLIPTokenizer

# Try to import torchvision, needed for FID. Fail gracefully if not available.
try:
    from torchvision import models as torchvision_models, transforms as torchvision_transforms
except ImportError:
    print("Warning: torchvision not found. FID calculation will be skipped if attempted.")
    torchvision_models, torchvision_transforms = None, None # Make them None so checks fail later

# 导入简化的PDU和SEU
from Immuno_Diffusion import PrivacyDetectionUnit, PrivacyEnhancementUnit 

print(f"--- Python Script Start ---")
# 先设置节点在终端$env:HF_ENDPOINT = "https://hf-mirror.com"
print(f"HF_ENDPOINT from os.environ: {os.getenv('HF_ENDPOINT')}")
print(f"--- End of ENV Check ---")

def parse_args():
    parser = argparse.ArgumentParser(description="Immuno-Diffusion Validation Script")
    # Modify dataset_path to be optional for hardcoded mode
    parser.add_argument("--dataset_path", type=str, default="internal_test_prompts", help="Path to the evaluation dataset or 'internal_test_prompts' for hardcoded prompts")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face model ID for the base diffusion model")
    parser.add_argument("--sensitive_keywords_path", type=str, help="Path to a file listing sensitive keywords for SLI calculation (also used by simplified PDU if no explicit list provided)")
    parser.add_argument("--pdu_sensitive_keywords", type=str, nargs='+', default=["secret", "private", "confidential", "internal"], help="List of sensitive keywords for the simplified PDU.")
    parser.add_argument("--seu_noise_level", type=float, default=0.05, help="Noise level for the simplified SEU.")
    parser.add_argument("--reference_fid_path", type=str, help="Path to the directory with reference images for FID. Required if --calculate_fid is set.")
    parser.add_argument("--output_dir", type=str, default="./validation_output_hardcoded", help="Directory to save generated images and results") # Changed default for hardcoded
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation.")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to generate (will be overridden if using internal_test_prompts)") # Default for general use
    parser.add_argument("--image_size", type=int, default=256, help="Size of generated images (height and width) - smaller for faster hardcoded test") # Smaller for faster test
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of DDIM inference steps - fewer for faster hardcoded test") # Fewer for faster test
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation") # Default seed for consistency
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--enable_privacy", action="store_true", help="Enable Immuno-Diffusion privacy mechanisms")
    parser.add_argument("--calculate_fid", action="store_true", help="Calculate FID score.")
    parser.add_argument("--calculate_sli", action="store_true", help="Calculate SLI score.")

    args = parser.parse_args([] if __name__ == "__main__" and os.getenv("RUNNING_AS_MAIN_SCRIPT_FOR_HARCODING") else None) # Parse no args if hardcoding intended for direct run
    
    if args.dataset_path != "internal_test_prompts" and not os.path.exists(args.dataset_path):
        parser.error(f"Dataset path {args.dataset_path} does not exist and is not 'internal_test_prompts'.")

    if args.calculate_fid and not args.reference_fid_path:
        parser.error("--reference_fid_path is required when --calculate_fid is set.")
    
    if args.enable_privacy: # Check PDU keywords only if privacy is enabled
        pdu_kws_provided = bool(args.pdu_sensitive_keywords and args.pdu_sensitive_keywords != ["secret", "private", "confidential", "internal"]) # Check if non-default
        loaded_from_file = False
        if not pdu_kws_provided and args.sensitive_keywords_path and os.path.exists(args.sensitive_keywords_path):
            print(f"Using sensitive keywords from {args.sensitive_keywords_path} for PDU as --pdu_sensitive_keywords was not set to a custom list.")
            with open(args.sensitive_keywords_path, 'r', encoding='utf-8') as f:
                args.pdu_sensitive_keywords = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            loaded_from_file = True
        
        if not args.pdu_sensitive_keywords:
            print("Warning: PDU is enabled but no sensitive keywords are defined. PDU might not detect any risk.")
            args.pdu_sensitive_keywords = []
        elif not loaded_from_file and pdu_kws_provided:
             print(f"PDU using custom keywords: {args.pdu_sensitive_keywords}")
        elif not loaded_from_file and not pdu_kws_provided:
             print(f"PDU using default keywords: {args.pdu_sensitive_keywords}")


    return args

def load_models(args):
    """Loads the necessary models."""
    print("Loading models...")
    device = torch.device(args.device)
    model_id = args.model_id

    try:
        # 这些调用会受到 HF_ENDPOINT 环境变量的影响
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        print(f"Base diffusion components ({model_id}) loaded successfully.")
    except Exception as e:
        print(f"Error loading base diffusion model from {model_id}: {e}")
        print("Please ensure the model_id is correct and you have an internet connection if downloading, or the model is cached.")
        raise

    pdu, seu = None, None
    if args.enable_privacy:
        print(f"Privacy mechanisms enabled. Initializing simplified PDU and SEU.")
        # PDU expects feature_dim for its dummy output, matching ImmuneMemoryModule input_dim (default 512)
        pdu = PrivacyDetectionUnit(sensitive_keywords=args.pdu_sensitive_keywords, device=args.device, feature_dim=512).to(device)
        
        seu_latent_dim = unet.config.in_channels
        seu = PrivacyEnhancementUnit(latent_dim=seu_latent_dim, simple_noise_level=args.seu_noise_level).to(device)
        print(f"Simplified PDU and SEU initialized. PDU keywords: {args.pdu_sensitive_keywords}, SEU noise: {args.seu_noise_level}.")
    else:
        print("Privacy mechanisms disabled.")

    clip_model, clip_processor, fid_metric = None, None, None 

    models = {
        "tokenizer": tokenizer, "text_encoder": text_encoder,
        "vae": vae, "unet": unet, "scheduler": scheduler,
        "pdu": pdu, "seu": seu, 
        "clip_model": clip_model, "clip_processor": clip_processor, 
        "fid_metric": fid_metric, 
        "device": device,
        "vae_scale_factor": 2 ** (len(vae.config.block_out_channels) - 1)
    }
    print("Models dictionary created.")
    return models

def load_dataset(args):
    """Loads the evaluation dataset. If args.dataset_path is 'internal_test_prompts', uses hardcoded prompts."""
    prompts_to_use = []
    sensitive_flags = [] # Corresponding flags for prompts
    sli_sensitive_keywords = [] # Keywords for SLI calculation, separate from PDU

    if args.dataset_path == "internal_test_prompts":
        print("Using internal hardcoded test prompts.")
        prompts_to_use = [
            "A photorealistic cat riding a unicorn on a rainbow",
            "A vibrant oil painting of a bustling Parisian cafe in the rain",
            "My secret project is about a confidential meeting for a new recipe" # Contains PDU default keywords
        ]
        # For hardcoded prompts, we can manually set sensitive_flags if needed for other logic,
        # but for basic generation, it's not critical.
        # Here, we mark the third one as potentially sensitive for demonstration if SLI were active.
        sensitive_flags = [False, False, True] 
        args.num_samples = len(prompts_to_use) # Ensure all hardcoded prompts are used
        print(f"Loaded {len(prompts_to_use)} hardcoded prompts.")
    else:
        print(f"Loading dataset from {args.dataset_path}...")
        prompts_data = []
        if args.dataset_path.endswith('.csv'):
            try:
                import pandas as pd
                df = pd.read_csv(args.dataset_path)
                if 'prompt' not in df.columns:
                    raise ValueError("CSV file must contain a 'prompt' column.")
                prompts_data = df['prompt'].astype(str).tolist()
            except ImportError:
                print("Pandas not installed, please install to load CSV datasets or use a .txt file.")
                raise
            except Exception as e:
                print(f"Error loading CSV: {e}")
                raise
        elif args.dataset_path.endswith('.txt'):
            try:
                with open(args.dataset_path, 'r', encoding='utf-8') as f:
                    prompts_data = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            except Exception as e:
                print(f"Error loading TXT file: {e}")
                raise
        else:
            print(f"Unsupported dataset file format: {args.dataset_path}. Please use .csv or .txt")
            # Fallback to dummy data if dataset loading fails for some reason or not specified properly
        
        if not prompts_data:
            print("No prompts loaded from dataset file, using fallback dummy prompts.")
            prompts_data = [f"Fallback sample prompt {i}" for i in range(args.num_samples)] 
        
        prompts_to_use = prompts_data[:args.num_samples]
        sensitive_flags = [i % 5 == 0 for i in range(len(prompts_to_use))] # Dummy flags for external datasets
        print(f"Loaded/selected {len(prompts_to_use)} prompts for generation from file.")

    # Load SLI sensitive keywords (distinct from PDU keywords, though can be from same file)
    if args.sensitive_keywords_path and os.path.exists(args.sensitive_keywords_path):
        with open(args.sensitive_keywords_path, 'r', encoding='utf-8') as f:
            sli_sensitive_keywords = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"Loaded {len(sli_sensitive_keywords)} sensitive keywords for SLI calculation.")
    elif args.calculate_sli:
        print("Warning: --calculate_sli is set but --sensitive_keywords_path was not provided or found for SLI.")

    return prompts_to_use, sensitive_flags, sli_sensitive_keywords

# ... prepare_text_input, prepare_graph_input remain placeholders ...

@torch.no_grad()
def generate_images(prompts, models, args):
    """Generates images using the base diffusion model, with optional simplified privacy modules."""
    if args.enable_privacy and models.get("pdu") and models.get("seu"):
        print(f"Generating images with simplified PDU/SEU privacy mechanisms enabled (SEU Noise: {args.seu_noise_level}).")
    else:
        print(f"Generating images using base diffusion model (privacy mechanisms disabled or not loaded).")
        
    device = models["device"]
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    vae = models["vae"]
    unet = models["unet"]
    scheduler = models["scheduler"]
    vae_scale_factor = models["vae_scale_factor"]

    pdu = models.get("pdu")
    seu = models.get("seu")

    height = args.image_size
    width = args.image_size
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    batch_size = args.batch_size

    # Use a generator for reproducible results if seed is provided
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    if args.seed is not None: # Also set global Pytorch seed
        torch.manual_seed(args.seed)
        if device == "cuda": torch.cuda.manual_seed_all(args.seed)

    generated_images_pil = []
    all_batch_risk_scores = [] # To collect PDU risk scores
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Generating Batches"):
        batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
        current_batch_size = len(batch_prompts)

        text_input = tokenizer(batch_prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

        if guidance_scale > 1.0:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = tokenizer([""] * current_batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents_shape = (current_batch_size, unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor)
        if generator is not None:
            latents = torch.randn(latents_shape, generator=generator, device=device, dtype=text_embeddings.dtype)
        else:
            latents = torch.randn(latents_shape, device=device, dtype=text_embeddings.dtype)
        latents = latents * scheduler.init_noise_sigma

        current_risk_score = None
        if args.enable_privacy and pdu:
            risk_score_batch, _ = pdu(batch_prompts) 
            current_risk_score = risk_score_batch.to(device)
            all_batch_risk_scores.append(current_risk_score.detach().cpu().numpy()) # Collect scores
            print(f"  Batch {i+1} PDU risk scores: {current_risk_score.squeeze().tolist()}") # Log risk scores

        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps

        for t_idx, t in enumerate(tqdm(timesteps, leave=False, desc="Denoising Steps")):
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            prev_latents = scheduler.step(noise_pred, t, latents).prev_sample

            if args.enable_privacy and seu and current_risk_score is not None:
                latents = seu(prev_latents, t, current_risk_score, memory_signal=None)
                # Optional: log if SEU is applied, e.g., for high risk prompts
                # if t_idx == 0 and current_risk_score.max() > 0.5:
                #     print(f"    SEU applied for batch {i+1} due to high risk.")
            else:
                latents = prev_latents

        latents = 1 / 0.18215 * latents 
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image_np = image.cpu().permute(0, 2, 3, 1).numpy()
        
        for img_np_single in image_np:
            pil_img = Image.fromarray((img_np_single * 255).astype(np.uint8))
            generated_images_pil.append(pil_img)
        
    print(f"Total generated images: {len(generated_images_pil)}")
    final_risk_scores = np.concatenate(all_batch_risk_scores) if all_batch_risk_scores else np.array([])
    return generated_images_pil, final_risk_scores

# --- FID Calculation Functions ---
def get_inception_model_for_fid(device):
    """Loads the InceptionV3 model for FID calculation."""
    if torchvision_models is None:
        raise ImportError("torchvision.models is not available.")
    inception_model = torchvision_models.inception_v3(weights=torchvision_models.Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
    inception_model.fc = torch.nn.Identity() # Use identity to get features
    inception_model.eval()
    return inception_model

def preprocess_image_for_fid(img_pil, device):
    """Preprocesses a PIL image for InceptionV3."""
    if torchvision_transforms is None:
        raise ImportError("torchvision.transforms is not available.")
    
    # Standard InceptionV3 preprocessing
    transform = torchvision_transforms.Compose([
        torchvision_transforms.Resize(299),
        torchvision_transforms.CenterCrop(299),
        torchvision_transforms.ToTensor(),
        torchvision_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    return transform(img_pil).unsqueeze(0).to(device)

def get_activations_for_fid(image_list_pil, model, batch_size, device):
    """Calculates InceptionV3 activations for a list of PIL images."""
    if not image_list_pil:
        return np.empty((0, 2048)) # InceptionV3 feature dimension

    num_batches = (len(image_list_pil) + batch_size - 1) // batch_size
    pred_arr = []

    for i in tqdm(range(num_batches), desc="Calculating Inception Activations"):
        batch_pil_images = image_list_pil[i * batch_size : (i + 1) * batch_size]
        if not batch_pil_images:
            continue

        batch_tensor = torch.cat([preprocess_image_for_fid(img, device) for img in batch_pil_images], dim=0)
        
        with torch.no_grad():
            pred = model(batch_tensor)

        if isinstance(pred, tuple): # InceptionV3 in training mode might return tuple
            pred = pred[0]
        
        pred_arr.append(pred.cpu().numpy().reshape(pred.size(0), -1))

    return np.concatenate(pred_arr, axis=0)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance using torch.linalg.sqrtm."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Covariance matrices have different dimensions'

    diff = mu1 - mu2

    # Convert numpy arrays to torch tensors for sqrtm
    # Ensure they are float64 for precision, and then potentially complex for sqrtm
    sigma1_torch = torch.from_numpy(sigma1).to(dtype=torch.complex128 if sigma1.dtype != np.complex128 else sigma1.dtype)
    sigma2_torch = torch.from_numpy(sigma2).to(dtype=torch.complex128 if sigma2.dtype != np.complex128 else sigma2.dtype)
    
    # Calculate (sigma1 @ sigma2)
    covmean_matrix_prod = sigma1_torch @ sigma2_torch
    
    # Calculate sqrt of the product of covariance matrices
    # torch.linalg.sqrtm can handle non-symmetric matrices and return complex results
    try:
        sqrt_covmean_matrix_prod = torch.linalg.sqrtm(covmean_matrix_prod)
    except Exception as e:
        print(f"torch.linalg.sqrtm failed: {e}. Using pseudo-inverse if applicable or failing.")
        # Fallback or error, FID might be unstable or NaN.
        # For now, re-raise or return NaN to indicate failure.
        # A more robust solution might involve checking condition numbers or using pseudo-sqrt.
        # However, a common cause is non-positive semi-definite product, often due to small sample sizes.
        raise ValueError(f"Matrix square root computation failed: {e}")


    # If the result is complex, take the real part if imaginary part is small
    if torch.is_complex(sqrt_covmean_matrix_prod):
        if torch.max(torch.abs(sqrt_covmean_matrix_prod.imag)) > 1e-3: # Tolerance
            print(f"Warning: Complex result from sqrtm with significant imaginary part ({torch.max(torch.abs(sqrt_covmean_matrix_prod.imag)):.2e}). FID might be unstable.")
        sqrt_covmean_matrix_prod = sqrt_covmean_matrix_prod.real # Take real part

    sqrt_covmean = sqrt_covmean_matrix_prod.numpy().astype(np.float64)
    
    tr_covmean = np.trace(sqrt_covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fid

def calculate_fid(generated_images_pil, reference_path, models_dict, args):
    if not args.calculate_fid:
        return float('nan')
    
    if torchvision_models is None or torchvision_transforms is None:
        print("torchvision.models or torchvision.transforms could not be imported. FID calculation skipped.")
        return float('nan')

    if not reference_path or not os.path.exists(reference_path):
        print(f"Reference FID path {reference_path} not found. Skipping FID.")
        return float('nan')
    
    print("Calculating FID...")
    device = models_dict["device"]
    
    try:
        inception_model = get_inception_model_for_fid(device)
    except Exception as e:
        print(f"Could not load InceptionV3 model for FID: {e}. Skipping FID.")
        print("Make sure torchvision is installed and can download pretrained models (check internet/HF_ENDPOINT).")
        return float('nan')

    # Load reference images
    ref_image_paths = [os.path.join(reference_path, f) for f in os.listdir(reference_path) 
                       if os.path.isfile(os.path.join(reference_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    if not ref_image_paths:
        print(f"No reference images found in {reference_path}. Skipping FID.")
        return float('nan')
        
    ref_images_pil = []
    for p in tqdm(ref_image_paths, desc="Loading reference FID images"):
        try:
            img = Image.open(p).convert('RGB')
            ref_images_pil.append(img)
        except Exception as e:
            print(f"Warning: Could not load reference image {p}: {e}")
    
    if not ref_images_pil:
        print(f"Failed to load any reference images. Skipping FID.")
        return float('nan')
    
    if not generated_images_pil:
        print("No generated images to calculate FID for. Skipping FID.")
        return float('nan')

    print(f"Calculating FID with {len(generated_images_pil)} generated images and {len(ref_images_pil)} reference images.")

    fid_batch_size = min(32, args.batch_size * 2 if hasattr(args, 'batch_size') else 32) # Make it robust
    
    try:
        act_generated = get_activations_for_fid(generated_images_pil, inception_model, fid_batch_size, device)
        act_reference = get_activations_for_fid(ref_images_pil, inception_model, fid_batch_size, device)
    except Exception as e:
        print(f"Error getting activations for FID: {e}. Skipping FID.")
        return float('nan')

    if act_generated.shape[0] < 2 or act_reference.shape[0] < 2: # Need at least 2 samples to compute covariance
        print(f"Not enough activations to compute FID (generated: {act_generated.shape[0]}, reference: {act_reference.shape[0]}). Need at least 2 for each. Skipping FID.")
        return float('nan')
        
    mu_gen, sigma_gen = np.mean(act_generated, axis=0), np.cov(act_generated, rowvar=False)
    mu_ref, sigma_ref = np.mean(act_reference, axis=0), np.cov(act_reference, rowvar=False)
    
    # Add epsilon to diagonal of covariances for stability if they are singular
    # This is a common practice if sample size is small relative to feature dimension
    epsilon = eps=1e-6
    sigma_gen += np.eye(sigma_gen.shape[0]) * epsilon
    sigma_ref += np.eye(sigma_ref.shape[0]) * epsilon

    try:
        fid_value = calculate_frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)
    except ValueError as e:
        print(f"Error calculating Frechet distance: {e}. This can happen with ill-conditioned covariance matrices (e.g. too few samples). Skipping FID.")
        return float('nan')
    except Exception as e:
        print(f"Unexpected error calculating Frechet distance: {e}. Skipping FID.")
        return float('nan')
        
    print(f"Calculated FID: {fid_value:.4f}")
    return float(fid_value)

def calculate_sli(generated_images_pil, prompts, sensitive_flags, sli_sensitive_keywords, models_dict, args):
    if not args.calculate_sli:
        return float('nan')
    
    if not sli_sensitive_keywords:
        print("SLI calculation active, but no SLI sensitive keywords provided. SLI will be 0 or NaN.")
        return 0.0 

    print(f"Calculating SLI with {len(sli_sensitive_keywords)} keywords (showing up to 5): {sli_sensitive_keywords[:5]}...")
    
    num_prompts_with_keywords = 0
    # This simple SLI counts how many sensitive prompts resulted in an image.
    # It doesn't analyze image content for actual leakage.
    num_sensitive_prompts_with_successful_generation = 0 

    if len(prompts) != len(generated_images_pil) and args.dataset_path == "internal_test_prompts":
         # This conditionality might be too specific for "internal_test_prompts"
         # A general warning is better if counts don't match for any reason
        print(f"Warning: Number of prompts ({len(prompts)}) and generated images ({len(generated_images_pil)}) mismatch. SLI might be based on fewer pairs than expected.")
    
    # Iterate up to the minimum length of prompts or generated images to avoid index errors
    num_comparisons = min(len(prompts), len(generated_images_pil))

    for i in range(num_comparisons):
        prompt_text = prompts[i]
        prompt_lower = prompt_text.lower()
        is_sensitive_prompt = False
        for keyword in sli_sensitive_keywords:
            if keyword.lower() in prompt_lower:
                is_sensitive_prompt = True
                break
        
        if is_sensitive_prompt:
            num_prompts_with_keywords += 1
            # Check if an image was actually generated for this sensitive prompt
            if generated_images_pil[i] is not None: # Assuming None if generation failed for a specific prompt
                 num_sensitive_prompts_with_successful_generation +=1

    if num_prompts_with_keywords == 0:
        print("No prompts contained SLI sensitive keywords after checking available pairs. SLI is 0 or N/A.")
        return 0.0 

    sli_score = num_sensitive_prompts_with_successful_generation / num_prompts_with_keywords
    print(f"SLI: {num_sensitive_prompts_with_successful_generation} successful generations for {num_prompts_with_keywords} sensitive prompts. SLI = {sli_score:.4f}")
    return float(sli_score)

def calculate_ard(all_risk_scores, args):
    """Calculates Average Risk Detection (ARD) based on PDU scores."""
    if not args.enable_privacy:
        print("ARD: Privacy not enabled, PDU risk scores not applicable.")
        return float('nan')
    
    if all_risk_scores is None or all_risk_scores.size == 0:
        if args.enable_privacy:
             print("ARD: Privacy enabled, but no PDU risk scores were collected (PDU might be misconfigured or no prompts processed).")
        return float('nan')
    
    # all_risk_scores should be a flat numpy array of scores for each item processed by PDU.
    average_risk = np.mean(all_risk_scores)
    print(f"Calculating ARD (Average PDU Risk Score over {all_risk_scores.size} items): {average_risk:.4f}")
    return float(average_risk)

def save_results(results, args):
    """Saves results and generated images."""
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")
    import json
    try:
        serializable_config = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in results["config"].items()}
        results_to_save = results.copy()
        results_to_save["config"] = serializable_config
        if "generated_images" in results_to_save: 
            del results_to_save["generated_images"]

        with open(results_path, "w") as f:
            json.dump(results_to_save, f, indent=4)
        print(f"Results saved to {results_path}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}. Saving as TXT instead.")
        results_path_txt = os.path.join(args.output_dir, "results.txt")
        with open(results_path_txt, "w") as f:
            for key, value in results.items():
                if key == "generated_images": continue 
                f.write(f"{key}: {value}\n")
        print(f"Results saved to {results_path_txt}")

    img_save_dir = os.path.join(args.output_dir, "images")
    os.makedirs(img_save_dir, exist_ok=True)
    images_pil_list = results.get("generated_images", [])
    num_save = min(len(images_pil_list), 16) 
    saved_paths = []
    if images_pil_list:
        for i in range(num_save):
            img = images_pil_list[i]
            if isinstance(img, Image.Image):
                try:
                    img_path = os.path.join(img_save_dir, f"sample_{i:03d}_{args.seed if args.seed else 'nseed'}_{('priv' if args.enable_privacy else 'nopriv')}.png")
                    img.save(img_path)
                    saved_paths.append(img_path)
                except Exception as e:
                    print(f"Error saving image sample_{i:03d}.png: {e}")
            else:
                print(f"Skipping saving image {i} as it is not a PIL Image object.")
        if saved_paths:
            print(f"Saved {len(saved_paths)} sample images to {img_save_dir}")
    results["generated_images_pil_paths"] = saved_paths


def main():
    # Set an environment variable to signal parse_args to use an empty list for sys.argv
    # This allows running the script directly with hardcoded defaults without needing CLI args.
    os.environ["RUNNING_AS_MAIN_SCRIPT_FOR_HARCODING"] = "1" 
    args = parse_args()
    del os.environ["RUNNING_AS_MAIN_SCRIPT_FOR_HARCODING"] # Clean up env var

    # --- Hardcode settings for a basic text-to-image run --- 
    args.dataset_path = "internal_test_prompts" # Ensure internal prompts are used
    args.enable_privacy = False # Disable privacy for the most basic run
    args.calculate_fid = False # Keep FID off for basic hardcoded run by default, can be overridden by CLI
    args.calculate_sli = False # Keep SLI off for basic hardcoded run
    # args.image_size = 256 # Already defaulted smaller in parse_args for quick test
    # args.num_inference_steps = 20 # Already defaulted fewer in parse_args
    # args.output_dir = "./validation_output_hardcoded_basic" # Can override if needed
    # args.seed = 42 # Already defaulted
    # --- End of hardcoded settings ---

    print("Starting validation with hardcoded basic settings...")
    print(f"Effective Arguments: {vars(args)}")

    models_dict = load_models(args) # Renamed to models_dict to avoid conflict if 'models' is imported from torchvision
    prompts, sensitive_flags, sli_keywords = load_dataset(args)
    
    generated_images_pil_list, all_risk_scores = generate_images(prompts, models_dict, args)

    fid_score = calculate_fid(generated_images_pil_list, args.reference_fid_path, models_dict, args)
    sli_score = calculate_sli(generated_images_pil_list, prompts, sensitive_flags, sli_keywords, models_dict, args)
    ard_score = calculate_ard(all_risk_scores, args)


    results = {
        "config": vars(args),
        "fid": fid_score,
        "sli": sli_score,
        "ard": ard_score, 
        "generated_images": generated_images_pil_list 
    }

    print("\n--- Validation Results ---")
    print(f"FID: {fid_score if not np.isnan(fid_score) else 'N/A'}")
    print(f"SLI: {sli_score if not np.isnan(sli_score) else 'N/A'}")
    print(f"ARD: {ard_score if not np.isnan(ard_score) else 'N/A'}")
    print("------------------------")

    save_results(results, args)
    print("Validation finished.")

if __name__ == "__main__":
    main() 