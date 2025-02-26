from datasets import load_dataset

from params import HF_DS_NAMES

if __name__ == "__main__":
    for ds_name in HF_DS_NAMES:
        print(f"Downloading {ds_name}")
        ds = load_dataset(ds_name)
        print(f"Downloaded {ds_name}")
