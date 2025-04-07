from huggingface_hub import snapshot_download

from gen_params import LLMS

if __name__ == "__main__":
    for model, path, quant in LLMS:
        print(f"Downloading {model}...")
        snapshot_download(
            repo_id=model,
            local_dir=path,
            local_dir_use_symlinks=False,
            ignore_patterns="*.pth",
        )
        print(f"Downloaded {model}!")
