import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def main():
    # 1. Setup paths
    image_dir = "data/images"
    embed_dir = "embeddings"
    os.makedirs(embed_dir, exist_ok=True)

    if not os.path.exists(image_dir) or len(os.listdir(image_dir)) == 0:
        print(f"Please put some .jpg images in the '{image_dir}' folder before running.")
        return

    # 2. Load CLIP Model
    print("Loading CLIP model...")
    model_id = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    # 3. Get image paths (limit to 5000 for quick demo)
    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = []
    for f in os.listdir(image_dir):
        if os.path.splitext(f)[1].lower() in valid_extensions:
            image_paths.append(os.path.join(image_dir, f))
    image_paths = image_paths[:5000]

    # 4. Extract embeddings
    print(f"Extracting features for {len(image_paths)} images on {device}...")
    embeddings = []
    valid_paths = []
    
    # Process in batches of 32
    batch_size = 32
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        current_valid_paths = []
        
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                current_valid_paths.append(p)
            except Exception as e:
                print(f"Skipping corrupt image {p}: {e}")
                
        if not images:
            continue
            
        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt").to(device)
            image_features = model.get_image_features(**inputs)
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output
            # L2 Normalization
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(image_features.cpu().numpy())
            valid_paths.extend(current_valid_paths)

    # 5. Save to disk
    if embeddings:
        embeddings_matrix = np.vstack(embeddings)
        np.save(os.path.join(embed_dir, "image_embeddings.npy"), embeddings_matrix)
        np.save(os.path.join(embed_dir, "image_paths.npy"), np.array(valid_paths))
        print("Embeddings successfully saved to 'embeddings/' folder.")
    else:
        print("No valid images found to process.")

if __name__ == "__main__":
    main()
