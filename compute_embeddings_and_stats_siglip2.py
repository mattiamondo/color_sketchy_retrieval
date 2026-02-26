import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import clip
from transformers import AutoModel, AutoProcessor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# 1. Load CLIP (ViT-B/32)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

# 2. Load SigLIP 2 
siglip2_ckpt = "google/siglip2-so400m-patch14-384"
siglip2_model = AutoModel.from_pretrained(siglip2_ckpt, device_map=DEVICE).eval()
siglip2_processor = AutoProcessor.from_pretrained(siglip2_ckpt, use_fast=True)

def get_file_size_mb(path: str) -> float:
    """Return file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)

def load_image(path: str):
    
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[warn] Failed to load image: {path} | {e}")
        return None

def encode_batch(images, texts, max_chars=400):
    """
    Encode a batch of images and texts using both CLIP and SigLIP 2.
    """
    # Basic text cleaning
    safe_texts = [str(t or "").strip()[:max_chars] for t in texts]

    with torch.no_grad():
        clip_image_input = torch.stack([clip_preprocess(img) for img in images]).to(DEVICE)
        clip_text_input = clip.tokenize(safe_texts, truncate=True).to(DEVICE)
        
        clip_img_feat = clip_model.encode_image(clip_image_input)
        clip_txt_feat = clip_model.encode_text(clip_text_input)
        
        # L2 Normalization
        clip_img_feat /= clip_img_feat.norm(dim=-1, keepdim=True)
        clip_txt_feat /= clip_txt_feat.norm(dim=-1, keepdim=True)

        siglip2_inputs = siglip2_processor(
            text=safe_texts, 
            images=images, 
            padding="max_length", 
            max_length=64, 
            truncation=True, 
            return_tensors="pt"
        ).to(DEVICE)


        siglip2_img_feat = siglip2_model.get_image_features(
            pixel_values=siglip2_inputs.get('pixel_values')
        )
        siglip2_txt_feat = siglip2_model.get_text_features(
            input_ids=siglip2_inputs.get('input_ids'),
            attention_mask=siglip2_inputs.get('attention_mask')
        )
        
        # L2 Normalization
        siglip2_img_feat /= siglip2_img_feat.norm(dim=-1, keepdim=True)
        siglip2_txt_feat /= siglip2_txt_feat.norm(dim=-1, keepdim=True)

    return (
        clip_img_feat.cpu().numpy(), 
        clip_txt_feat.cpu().numpy(), 
        siglip2_img_feat.cpu().numpy(), 
        siglip2_txt_feat.cpu().numpy()
    )

def process_dataset(entries, image_key: str, text_key: str, output_dir: str, batch_size=32):
    """
    Process the entire dataset and save embeddings to .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_clip_img, all_clip_txt = [], []
    all_sig2_img, all_sig2_txt = [], []
    
    total_image_size_mb = 0.0
    missing_or_bad = 0
    valid_images, valid_texts = [], []

    # 1. Load data into memory
    for e in tqdm(entries, desc=f"Loading images from {output_dir}"):
        img_path = e.get(image_key, "")
        caption = str(e.get(text_key, "") or "")

        if not img_path or not os.path.exists(img_path):
            missing_or_bad += 1
            continue

        img = load_image(img_path)
        if img is None:
            missing_or_bad += 1
            continue

        valid_images.append(img)
        valid_texts.append(caption)
        total_image_size_mb += get_file_size_mb(img_path)

    # 2. Batch encoding to prevent OOM
    num_samples = len(valid_images)
    for i in tqdm(range(0, num_samples, batch_size), desc=f"Encoding {output_dir}"):
        batch_imgs = valid_images[i : i + batch_size]
        batch_txts = valid_texts[i : i + batch_size]
        
        c_img, c_txt, s2_img, s2_txt = encode_batch(batch_imgs, batch_txts)
        
        all_clip_img.append(c_img)
        all_clip_txt.append(c_txt)
        all_sig2_img.append(s2_img)
        all_sig2_txt.append(s2_txt)

    # 3. Concatenate and save
    if num_samples > 0:
        final_clip_img = np.concatenate(all_clip_img, axis=0)
        final_clip_txt = np.concatenate(all_clip_txt, axis=0)
        final_sig2_img = np.concatenate(all_sig2_img, axis=0)
        final_sig2_txt = np.concatenate(all_sig2_txt, axis=0)
    else:
        # Fallback for empty datasets
        final_clip_img = np.zeros((0, 512), dtype=np.float32)
        final_clip_txt = np.zeros((0, 512), dtype=np.float32)
        final_sig2_img = np.zeros((0, 1152), dtype=np.float32)
        final_sig2_txt = np.zeros((0, 1152), dtype=np.float32)

    np.save(os.path.join(output_dir, "clip_image.npy"), final_clip_img)
    np.save(os.path.join(output_dir, "clip_text.npy"), final_clip_txt)
    np.save(os.path.join(output_dir, "siglip2_image.npy"), final_sig2_img)
    np.save(os.path.join(output_dir, "siglip2_text.npy"), final_sig2_txt)

    return {
        "num_images": int(num_samples),
        "missing_or_bad": int(missing_or_bad),
        "total_image_size_mb": float(total_image_size_mb),
        "clip_dim": int(final_clip_img.shape[1]),
        "siglip2_dim": int(final_sig2_img.shape[1]),
        "siglip2_total_embedding_mb": float((final_sig2_img.nbytes + final_sig2_txt.nbytes) / (1024 * 1024))
    }


def main():
    stats = {}

    # 1. Color Dataset
    color_metadata_path = "data/color/metadata.json"
    color_root = "data/color"

    if os.path.exists(color_metadata_path):
        with open(color_metadata_path, "r") as f:
            color_entries = json.load(f)
        for e in color_entries:
            e["abs_image"] = os.path.join(color_root, e["image"])
            e["caption"] = e.get("name", "")

        stats["color"] = process_dataset(
            color_entries, "abs_image", "caption", "embeddings/color"
        )

    #  2. Sketchy Test
    sketchy_test_path = "data/sketchy_test/metadata.json"
    if os.path.exists(sketchy_test_path):
        with open(sketchy_test_path, "r") as f:
            sketchy_test_entries = json.load(f)

        stats["sketchy_test"] = process_dataset(
            sketchy_test_entries, "original_filename", "original_caption", "embeddings/sketchy_test"
        )

    #  3. Sketchy Train (Commented out by default) 
    # sketchy_train_path = "/mnt/data/zruan/workspace_novel/zruan/GOAL/datasets/SKETCHY_train_nested_with_sketches.json"
    # if os.path.exists(sketchy_train_path):
    #     with open(sketchy_train_path, "r") as f:
    #         sketchy_train_entries = json.load(f)
    #     stats["sketchy_train"] = process_dataset(
    #         sketchy_train_entries, "original_filename", "original_caption", "outputs/sketchy_train"
    #     )

    # Save summary stats
    with open("embeddings/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\nProcessing complete! Embeddings saved in outputs/ directory. Stats saved in embeddings/stats.json")

if __name__ == "__main__":
    main()