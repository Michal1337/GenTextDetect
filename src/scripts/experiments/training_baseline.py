import argparse
from torch.utils.data import DataLoader
from ex_params import DATASETS_PATH
from ex_utils import TextDataset, collate_fn


def train(model, dataloader, epochs, save_path, logs_path):
    pass


if __name__ == "__main__":
    # Command-line interface
    parser = argparse.ArgumentParser(
        description="Train the baseline model."
    )
    parser.add_argument("model_size", type=str, help="Size of the model")
    parser.add_argument("dataset_size", type=str, help="Size of the dataset")
    parser.add_argument("epochs", type=int, help="Number of epochs")
    parser.add_argument("batch_size", type=int, help="Batch size")

    args = parser.parse_args()

    ds_path = f"{DATASETS_PATH}/{args.model_size}/dataset.csv" 
    df_data = pd.read_csv(ds_path)
    dataset = TextDataset(df_data["text"], df_data["label"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)


    