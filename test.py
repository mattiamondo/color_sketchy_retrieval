import torch
import numpy as np
from transformers import AutoModel, AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# USA GLI STESSI IDENTICI PARAMETRI
siglip2_ckpt = "google/siglip2-so400m-patch14-384"
siglip2_model = AutoModel.from_pretrained(siglip2_ckpt, device_map=DEVICE).eval()
siglip2_processor = AutoProcessor.from_pretrained(siglip2_ckpt, use_fast=True)

# Carica embedding pre-calcolati
image_embeddings = np.load("embeddings/sketchy_test/siglip2_image.npy")

def retrieve(query_text: str, top_k: int = 5):
    """Retrieval usando la STESSA procedura del codice originale"""
    
    # STESSA procedura: max_length=64, padding="max_length"
    inputs = siglip2_processor(
        text=[query_text],  # Lista anche per singola query
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        text_feat = siglip2_model.get_text_features(
            input_ids=inputs.get('input_ids'),
            attention_mask=inputs.get('attention_mask')
        )
        
        # STESSA normalizzazione L2
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
    
    text_emb = text_feat.cpu().numpy()
    
    # Calcola similarità (dot product, già normalizzato)
    similarities = image_embeddings @ text_emb.T
    top_indices = np.argsort(similarities.squeeze())[::-1][:top_k]
    
    return top_indices, similarities.squeeze()[top_indices]

# Test
indices, scores = retrieve("red dress", top_k=5)
print(f"Top 5 matches: {indices}")
print(f"Scores: {scores}")