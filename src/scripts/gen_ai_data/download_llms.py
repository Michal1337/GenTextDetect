from gen_params import LLMS
from huggingface_hub import snapshot_download


if __name__ == "__main__":
    for model, quant, path in LLMS:
        print(f"Downloading {model}...")
        snapshot_download(
            repo_id=model,
            local_dir=path,
            local_dir_use_symlinks=False,
            ignore_patterns="*.pth",
        )
        print(f"Downloaded {model}!")
