import os

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

from ex_params import DATASETS_PATH, PREDICTIONS_PATH

BATCH_SIZE = 64


class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # Initialize the base transformer model.
        self.model = AutoModel.from_config(config)
        # Define a classifier head.
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights (handled by PreTrainedModel)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the transformer
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        # Mean pooling
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Classifier
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output


def predict_multiple_texts(texts, model, tokenizer, device, max_len=768, threshold=0.5):
    # Tokenize all texts
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probabilities = torch.sigmoid(logits).squeeze(-1)  # Shape: [batch_size]

    # Convert to Python lists for easier processing outside
    return probabilities.cpu().tolist()


def batchify(data_list, batch_size):
    """
    Splits a list into batches of a specified size.

    Args:
        data_list (list): The list to be split.
        batch_size (int): The maximum size of each batch.

    Returns:
        list of lists: A list where each sublist is a batch of elements.
    """
    return [data_list[i : i + batch_size] for i in range(0, len(data_list), batch_size)]


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("desklib/ai-text-detector-v1.01")
    # model = AutoModel.from_pretrained("desklib/ai-text-detector-v1.01")

    model_directory = "desklib/ai-text-detector-v1.01"

    # --- Load tokenizer and model ---
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = DesklibAIDetectionModel.from_pretrained(model_directory)

    # --- Set up device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    df_test = pd.read_csv(os.path.join(DATASETS_PATH, "master-testset/test.csv"))
    texts = df_test["text"].tolist()

    texts_batches = batchify(texts, batch_size=BATCH_SIZE)
    predictions = []
    for batch in tqdm(texts_batches):
        batch_predictions = predict_multiple_texts(batch, model, tokenizer, device)
        predictions.extend(batch_predictions)

    df_preds = pd.DataFrame(predictions, columns=["preds"])
    df_preds["data"] = df_test["data"]
    df_preds["model"] = df_test["model"]
    df_preds["label"] = df_test["label"]

    df_preds.to_csv(
        os.path.join(
            PREDICTIONS_PATH,
            "external_model/preds_master-testset.csv",
        ),
        index=False,
    )

    for level in range(6):
        df_test = pd.read_csv(
            os.path.join(DATASETS_PATH, f"master-testset-hard/test{level}.csv")
        )
        texts = df_test["text"].tolist()
        texts_batches = batchify(texts, batch_size=BATCH_SIZE)
        predictions = []
        for batch in tqdm(texts_batches):
            batch_predictions = predict_multiple_texts(batch, model, tokenizer, device)
            predictions.extend(batch_predictions)
        df_preds = pd.DataFrame(predictions, columns=["preds"])
        df_preds["data"] = df_test["data"]
        df_preds["model"] = df_test["model"]
        df_preds["label"] = df_test["label"]
        df_preds.to_csv(
            os.path.join(
                PREDICTIONS_PATH,
                f"external_model/preds_master-testset-hard-{level}.csv",
            ),
            index=False,
        )
