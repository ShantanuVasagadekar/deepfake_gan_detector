import os
import argparse
from huggingface_hub import snapshot_download

MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
SAVE_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "hf_deepfake_model")

def download_model(force=False):
    print(f"--- Deepfake Detector Downloader ---")
    print(f"Target: {MODEL_ID}")
    print(f"Local Path: {SAVE_PATH}")
    
    if os.path.exists(os.path.join(SAVE_PATH, "config.json")) and not force:
        print("\nModel already exists locally. Skipping download.")
        return

    print("\nDownloading model files from Hugging Face...")
    try:
        # Download the entire repository to a local folder
        # We use local_dir_use_symlinks=False to ensure files are actually in the directory
        download_path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=SAVE_PATH,
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN") # Uses token if available
        )
        print(f"\nSUCCESS: Model downloaded to: {download_path}")
        print("You can now run detect_image.py without an internet connection.")
    except Exception as e:
        print(f"\nERROR: Failed to download model: {e}")
        print("Check your internet connection or HF_TOKEN if the repository is private.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    args = parser.parse_args()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    download_model(force=args.force)
