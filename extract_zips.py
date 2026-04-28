import os
import zipfile
from pathlib import Path

DATASET_DIR = Path("dataset")

def extract_existing_zips():
    """Find and extract any .zip files that were downloaded but not extracted."""
    print("\n[EXTRACT] Checking for unextracted ZIP files...")
    zips = list(DATASET_DIR.rglob("*.zip"))
    
    if not zips:
        print("No ZIP files found.")
        return
        
    for zip_path in zips:
        print(f"  Extracting {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(zip_path.parent)
            print(f"  [DONE] Extracted {zip_path.name}")
        except Exception as e:
            print(f"  [ERROR] Could not extract {zip_path.name}: {e}")

if __name__ == "__main__":
    extract_existing_zips()
