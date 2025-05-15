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
    return generated_images_pil

def calculate_fid(generated_images_pil, reference_path, models, args):
    if not args.calculate_fid:
        # print("FID calculation skipped by args.")
        return float('nan')
    if not reference_path or not os.path.exists(reference_path):
        print(f"Reference FID path {reference_path} not found. Skipping FID.")
        return float('nan')
    print("Calculating FID (Placeholder)...")
    return 50.0 

def calculate_sli(generated_images_pil, prompts, sensitive_flags, sensitive_keywords, models, args):
    if not args.calculate_sli:
        # print("SLI calculation skipped by args.")
        return float('nan')
    print("Calculating SLI (Placeholder)...")
    return 0.1

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
    args.calculate_fid = False
    args.calculate_sli = False
    # args.image_size = 256 # Already defaulted smaller in parse_args for quick test
    # args.num_inference_steps = 20 # Already defaulted fewer in parse_args
    # args.output_dir = "./validation_output_hardcoded_basic" # Can override if needed
    # args.seed = 42 # Already defaulted
    # --- End of hardcoded settings ---

    print("Starting validation with hardcoded basic settings...")
    print(f"Effective Arguments: {vars(args)}")

    models = load_models(args)
    prompts, sensitive_flags, sli_keywords = load_dataset(args)
    
    generated_images_pil_list = generate_images(prompts, models, args)

    fid_score = calculate_fid(generated_images_pil_list, args.reference_fid_path, models, args)
    sli_score = calculate_sli(generated_images_pil_list, prompts, sensitive_flags, sli_keywords, models, args)

    results = {
        "config": vars(args),
        "fid": fid_score,
        "sli": sli_score,
        "generated_images": generated_images_pil_list 
    }

    print("\n--- Validation Results ---")
    print(f"FID: {fid_score if not np.isnan(fid_score) else 'N/A'}")
    print(f"SLI: {sli_score if not np.isnan(sli_score) else 'N/A'}")
    print("------------------------")

    save_results(results, args)
    print("Validation finished.")

if __name__ == "__main__":
    main() 