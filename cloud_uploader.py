import os
from lightning_sdk.client import Client
from lightning_sdk import Studio

# Set environment variables so the SDK picks them up
os.environ["LIGHTNING_USER_ID"] = "432c4b70-2a94-4392-9942-c4547938c39a"
os.environ["LIGHTNING_API_KEY"] = "db4afadd-2af6-415f-a79e-1aac972c50dc"

def run_upload():
    print("Authenticating with Lightning AI API...")
    try:
        # Create a blank studio specifically for this 8-GPU training
        print("Creating an empty PyTorch Cloud Studio...")
        studio = Studio(name="Deepfake-GPU-Training")
        
        print(f"Studio '{studio.name}' created successfully!")
        
        print("Uploading deepfake_upload.tar directly to the cloud...")
        # upload_file takes (source_path, target_path_in_studio)
        studio.upload_file("deepfake_upload.tar", "/home/zeus/deepfake_upload.tar")
        print("Upload complete!")
        print("-" * 50)
        print("SUCCESS. Please go back to lightning.ai in your browser.")
        print(f"You will find a new Studio named '{studio.name}'. Open it, and run:")
        print("    tar -xf deepfake_upload.tar")
        print("    python train_lightning.py")
    except Exception as e:
        print(f"Failed via API: {e}")

if __name__ == "__main__":
    run_upload()
