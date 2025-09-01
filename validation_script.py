import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import clip
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import subprocess
import re
import logging
from diffusers import StableDiffusionPipeline


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration and Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Immuno-Diffusion Validation Script")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the CSV file containing prompts and flags for evaluation.")
    parser.add_argument("--reference_fid_path", type=str, default=None,
                        help="Path to the folder containing reference images for FID calculation.")
    parser.add_argument("--output_dir", type=str, default="./validation_results",
                        help="Directory to save validation results and generated images.")
    parser.add_argument("--num_generations", type=int, default=100,
                        help="Number of images to generate for evaluation.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for image generation and evaluation.")
    parser.add_argument("--enable_privacy", action="store_true",
                        help="Enable privacy evaluation (SLI and ARD).")
    parser.add_argument("--calculate_fid", action="store_true",
                        help="Calculate Frechet Inception Distance (FID). Requires --reference_fid_path.")
    parser.add_argument("--calculate_sli", action="store_true",
                        help="Calculate Semantic Leakage Index (SLI). Requires CLIP model.")
    parser.add_argument("--calculate_ard", action="store_true",
                        help="Calculate Attribute Removal Distance (ARD). Requires BERT model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for computations (cuda or cpu).")
    parser.add_argument("--model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Path or Hugging Face name of the Stable Diffusion model.")
    parser.add_argument("--clip_model_name", type=str, default="ViT-B/32",
                        help="Name of the CLIP model to use for SLI.")
    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased",
                        help="Name of the BERT model to use for ARD.")
    # New: configurable torch home for pytorch-fid Inception cache
    parser.add_argument("--torch_home", type=str, default=None,
                        help="Folder for torch hub/checkpoints. If not set, will use <output_dir>/torch_checkpoints")
    return parser.parse_args()


# --- Model Loading ---
# --- Model Loading ---
def load_models(args):
    device = args.device
    logging.info(f"Loading models on device: {device}")

    # Load Stable Diffusion Pipeline
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
        pipeline = pipeline.to(device)
        logging.info(f"Stable Diffusion model '{args.model_name_or_path}' loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load Stable Diffusion model: {e}. Exiting.")
        exit()

    # Load other models for metrics
    clip_model, preprocess = None, None
    if args.calculate_sli:
        try:
            clip_model, preprocess = clip.load(args.clip_model_name, device=device, jit=False)
            clip_model.float()
            logging.info(f"CLIP model '{args.clip_model_name}' loaded.")
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}. SLI calculation will be skipped.")
            args.calculate_sli = False

    tokenizer, bert_model = None, None
    if args.calculate_ard:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
            bert_model = AutoModelForSequenceClassification.from_pretrained(args.bert_model_name).to(device)
            bert_model.half()
            logging.info(f"BERT model '{args.bert_model_name}' loaded.")
        except Exception as e:
            logging.error(f"Failed to load BERT model or tokenizer: {e}. ARD calculation will be skipped.")
            args.calculate_ard = False

    # 注意：返回的内容变了，不再有 immuno_diffusion_model 和 memory
    return pipeline, clip_model, preprocess, tokenizer, bert_model

# --- Image Generation ---
# --- Image Generation ---
def image_generations(pipeline, prompts, device, num_generations, clip_model=None):
    generated_images = []
    all_text_embeddings = []
    
    # Ensure we have enough prompts to generate the desired number of images
    all_prompts = prompts * (num_generations // len(prompts)) + prompts[:num_generations % len(prompts)]

    for prompt in tqdm(all_prompts, desc="Generating Images"):
        with torch.no_grad():
            # Generate one image using the pipeline
            image_pil = pipeline(prompt).images[0]
            
            # Convert PIL image to tensor for storage and evaluation
            img_tensor = transforms.ToTensor()(image_pil)
            generated_images.append(img_tensor)

            # If calculating SLI, we also need the text embeddings for the prompt
            if clip_model:
                text_inputs = clip.tokenize([prompt]).to(device)
                text_embedding = clip_model.encode_text(text_inputs)
                all_text_embeddings.append(text_embedding.cpu())

    # Stack all tensors into a single batch
    generated_images_tensor = torch.stack(generated_images)
    original_text_embeddings_tensor = torch.cat(all_text_embeddings) if all_text_embeddings else None
    
    return generated_images_tensor, original_text_embeddings_tensor
# --- Evaluation Metrics ---
def calculate_fid_score(real_images_path, generated_images_tensor, device, output_dir, torch_home):
    if not real_images_path:
        logging.warning("Reference FID path not provided. Skipping FID calculation.")
        return None

    # Save generated images to a temporary directory for FID calculation (inside output_dir)
    temp_gen_dir = os.path.join(output_dir, "temp_generated_images")
    os.makedirs(temp_gen_dir, exist_ok=True)
    for i, img_tensor in enumerate(generated_images_tensor):
        img = transforms.ToPILImage()(img_tensor)
        img.save(os.path.join(temp_gen_dir, f"gen_{i:05d}.png"))

    logging.info(f"Calculating FID between {real_images_path} and {temp_gen_dir} using pytorch_fid API...")
    try:
        # Configure TORCH_HOME for pytorch-fid model cache (Inception)
        if torch_home is None:
            torch_home = os.path.join(output_dir, "torch_checkpoints")
        os.makedirs(os.path.join(torch_home, "hub", "checkpoints"), exist_ok=True)
        os.environ["TORCH_HOME"] = torch_home

        from pytorch_fid.fid_score import calculate_fid_given_paths

        paths = [real_images_path, temp_gen_dir]
        # Use a reasonable batch size; fall back to 32 if args.batch_size is too small/heavy (handled by caller's arg)
        fid_score = calculate_fid_given_paths(paths, 32, device, 2048)
        logging.info(f"Calculated FID score: {fid_score}")
        return fid_score

    except ImportError:
        logging.error("`pytorch_fid` Python API not found. Please ensure `pytorch-fid` is installed (`pip install pytorch-fid`).")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during FID calculation using API: {e}")
        return None
    finally:
        # Clean temp images
        try:
            for f in os.listdir(temp_gen_dir):
                os.remove(os.path.join(temp_gen_dir, f))
            os.rmdir(temp_gen_dir)
        except Exception as _:
            pass
        if 'TORCH_HOME' in os.environ:
            del os.environ['TORCH_HOME']

def calculate_sli(generated_images_tensor, original_text_embeddings, clip_model, preprocess, device, batch_size=32):
    if not clip_model:
        logging.warning("CLIP model not loaded. Skipping SLI calculation.")
        return None

    logging.info("Calculating Semantic Leakage Index (SLI)...")
    image_features_list = []
    clip_model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(generated_images_tensor), batch_size), desc="Processing images for SLI"):
            batch_images = generated_images_tensor[i:i+batch_size]
            processed_images = []
            for img_tensor in batch_images:
                img_pil = transforms.ToPILImage()(img_tensor.cpu())
                processed_images.append(preprocess(img_pil))
            processed_images_batch = torch.stack(processed_images).to(device).float()
            image_features = clip_model.encode_image(processed_images_batch).float()
            image_features_list.append(image_features.cpu())

    all_image_features = torch.cat(image_features_list)
    all_image_features = F.normalize(all_image_features, p=2, dim=-1)
    original_text_embeddings = F.normalize(original_text_embeddings.float(), p=2, dim=-1)

    similarities = torch.sum(all_image_features * original_text_embeddings, dim=1)
    sli_score = similarities.mean().item()
    logging.info(f"Calculated SLI score: {sli_score}")
    return sli_score

def calculate_ard(generated_images_tensor, prompts, tokenizer, bert_model, device, batch_size=32):
    if not tokenizer or not bert_model:
        logging.warning("BERT model or tokenizer not loaded. Skipping ARD calculation.")
        return None

    logging.info("Calculating Attribute Removal Distance (ARD)...")
    bert_model.eval()
    bert_model.half()

    attribute_scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing prompts for ARD"):
            batch_prompts = prompts[i:i+batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            inputs['input_ids'] = inputs['input_ids'].long()
            inputs['attention_mask'] = inputs['attention_mask'].float()

            outputs = bert_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            attribute_scores.extend(probs[:, 0].cpu().tolist())

    ard_score = np.mean(attribute_scores) if attribute_scores else 0.0
    logging.info(f"Calculated (placeholder) ARD score: {ard_score}")
    return ard_score

# --- Main Validation Function ---
def run_validation(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Validation results will be saved to: {args.output_dir}")

    pipeline, clip_model, preprocess, tokenizer, bert_model = load_models(args)

    try:
        df = pd.read_csv(args.dataset_path)
        prompts = df["prompt"].tolist()
        logging.info(f"Loaded {len(prompts)} prompts from {args.dataset_path}")
    except FileNotFoundError:
        logging.error(f"Dataset CSV not found at {args.dataset_path}. Exiting.")
        return
    except KeyError:
        logging.error(f"CSV file {args.dataset_path} must contain a 'prompt' column. Exiting.")
        return

    logging.info(f"Generating {args.num_generations} images...")
    generated_images, original_text_embeddings = image_generations(
        pipeline, prompts, args.device, args.num_generations, clip_model
    )
    logging.info(f"Generated images tensor shape: {generated_images.shape}")

    gen_image_dir = os.path.join(args.output_dir, "generated_images")
    os.makedirs(gen_image_dir, exist_ok=True)
    for i, img_tensor in enumerate(generated_images):
        img = transforms.ToPILImage()(img_tensor)
        img.save(os.path.join(gen_image_dir, f"generated_{i:05d}.png"))
    logging.info(f"Saved {len(generated_images)} generated images to {gen_image_dir}")

    results = {}

    if args.calculate_fid:
        fid_score = calculate_fid_score(args.reference_fid_path, generated_images, args.device, args.output_dir, args.torch_home)
        if fid_score is not None:
            results["FID"] = fid_score
    else:
        logging.info("FID calculation skipped.")

    if args.calculate_sli:
        sli_score = calculate_sli(generated_images, original_text_embeddings, clip_model, preprocess, args.device, args.batch_size)
        if sli_score is not None:
            results["SLI"] = sli_score
    else:
        logging.info("SLI calculation skipped.")

    if args.calculate_ard:
        ard_score = calculate_ard(generated_images, prompts, tokenizer, bert_model, args.device, args.batch_size)
        if ard_score is not None:
            results["ARD"] = ard_score
    else:
        logging.info("ARD calculation skipped.")

    results_path = os.path.join(args.output_dir, "validation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Validation results saved to {results_path}")
    logging.info("Validation complete.")
    print("\n--- Validation Results ---")
    for metric, value in results.items():
        try:
            print(f"{metric}: {float(value):.4f}")
        except Exception:
            print(f"{metric}: {value}")
    print("--------------------------")

if __name__ == "__main__":
    args = parse_args()
    run_validation(args)