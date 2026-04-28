import os
from huggingface_hub import hf_hub_download

def download_xade():
    repo_id = "viktorahnstrom/xade-deepfake-detector"
    filename = "best_model.pt"
    local_dir = "saved_models/xade_model"
    
    print(f"Downloading {filename} from {repo_id}...")
    
    os.makedirs(local_dir, exist_ok=True)
    
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    
    print(f"Successfully downloaded to: {path}")
    return path

if __name__ == "__main__":
    download_xade()
