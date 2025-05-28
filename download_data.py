from huggingface_hub import list_repo_files, hf_hub_download
import os

# Repo info
repo_id = "rjgpinel/GEMBench"
repo_type = "dataset"
target_dir = "./GEMBench"

# List all files in the dataset repository
files = list_repo_files(repo_id=repo_id, repo_type=repo_type)

# Create directory if needed
os.makedirs(target_dir, exist_ok=True)

# Download all files
for file in files:
    print(f"Downloading: {file}")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=file,
        repo_type=repo_type,
        cache_dir=target_dir
    )
    print(f"Saved to: {local_path}")

print("✅ All files downloaded.")