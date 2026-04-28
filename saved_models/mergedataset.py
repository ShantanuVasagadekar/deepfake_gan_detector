import os
import shutil

# YOUR EXACT SOURCE PATH
source_root = r"C:\Users\Shantanu Vasagadekar\Downloads\archive\real_vs_fake\real-vs-fake"

# TARGET DATASET
target_root = "dataset"

real_target = os.path.join(target_root, "real")
fake_target = os.path.join(target_root, "fake", "human_faces")

os.makedirs(real_target, exist_ok=True)
os.makedirs(fake_target, exist_ok=True)

splits = ["train", "valid", "test"]

def copy_images(src, dst):
    count = 0
    for root, _, files in os.walk(src):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                src_file = os.path.join(root, file)

                # avoid duplicate overwrite
                dst_file = os.path.join(dst, file)
                if os.path.exists(dst_file):
                    base, ext = os.path.splitext(file)
                    dst_file = os.path.join(dst, f"{base}_{count}{ext}")

                shutil.copy2(src_file, dst_file)
                count += 1
    return count

total_real = 0
total_fake = 0

for split in splits:
    real_path = os.path.join(source_root, split, "real")
    fake_path = os.path.join(source_root, split, "fake")

    if os.path.exists(real_path):
        total_real += copy_images(real_path, real_target)

    if os.path.exists(fake_path):
        total_fake += copy_images(fake_path, fake_target)

print("Done merging!")
print(f"Total real images: {total_real}")
print(f"Total fake images: {total_fake}")