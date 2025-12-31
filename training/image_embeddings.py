import os
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import torch
import torchvision.transforms as T
import torchvision.models as models

# Add parent directory to path to find config/utils
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import DRIVE_CREDENTIALS_PATH, DRIVE_FOLDER_ID
from drive_utils_fast import get_images_batch_from_drive, init_fast_drive_loader

# --- Device & Model Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

for p in model.parameters():
    p.requires_grad = False

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#print(DRIVE_CREDENTIALS_PATH, DRIVE_FOLDER_ID)

# --- Drive Setup ---
try:
    init_fast_drive_loader(
        credentials_path=DRIVE_CREDENTIALS_PATH, 
        folder_id=DRIVE_FOLDER_ID
    )
except Exception as e:
    print(f"Failed to initialize Drive loader: {e}")
    sys.exit(1)

# --- Data Processing ---
df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/train.csv"))

def extract_embedding(image_link):
    try:
        # 1. Fetch from Drive using the link directly
        # The loader handles basename extraction internally
        result_dict = get_images_batch_from_drive([image_link])
        
        # 2. Retrieve the image using the link as key
        img = result_dict.get(image_link)
        
        if img is None:
            return None
            
        # 3. Process
        img = img.convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emb = model(img).cpu().numpy().flatten()
        return emb
    except Exception as e:
        return None

embeddings = []
valid_idx = []

print("Extracting embeddings...")
for img_link in tqdm(df['image_link'], desc="Loading Drive Images"):
    # CORRECTED: Pass the link directly. Do not use os.path.join(IMAGE_DIR, ...)
    print(f"Processing image link: {img_link}")
    emb = extract_embedding(img_link)
    if emb is not None:
        embeddings.append(emb)
        valid_idx.append(img_link)

# --- Save Results ---
if embeddings:
    X_img = np.vstack(embeddings)
    df_img = df.loc[valid_idx].reset_index(drop=True)

    np.save("image_embeddings.npy", X_img)
    df_img.to_csv("train_with_images.csv", index=False)
    print(f"Successfully processed {len(embeddings)} images.")
else:
    print("No embeddings were generated.")
