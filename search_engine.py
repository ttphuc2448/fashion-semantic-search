import os
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

class SemanticSearchEngine:
    def __init__(self, model_id="openai/clip-vit-base-patch32", embed_dir="embeddings"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        
        # Load pre-computed embeddings
        embed_path = os.path.join(embed_dir, "image_embeddings.npy")
        paths_path = os.path.join(embed_dir, "image_paths.npy")
        
        if not os.path.exists(embed_path) or not os.path.exists(paths_path):
            raise FileNotFoundError("Embeddings not found. Please run encode_data.py first.")
            
        self.image_embeddings = np.load(embed_path)
        self.image_paths = np.load(paths_path)

    def search(self, query: str, top_k: int = 5):
        with torch.no_grad():
            # Encode Text
            inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            if hasattr(text_features, 'pooler_output'):
                text_features = text_features.pooler_output
            
            # L2 Normalization
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy()

        # Compute Cosine Similarity (Dot product since both vectors are L2 normalized)
        similarities = np.dot(text_features, self.image_embeddings.T)[0]
        
        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "path": self.image_paths[idx],
                "score": float(similarities[idx])
            })
            
        return results
