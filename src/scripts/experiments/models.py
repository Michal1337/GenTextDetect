import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM


class FineTuneClassifier(nn.Module):
    def __init__(self, base_model_path: str, num_labels: int) -> None:
        super(FineTuneClassifier, self).__init__()
        self.base_model = AutoModel.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.base_model.config.hidden_size * 2, num_labels)

    @classmethod
    def from_classifier_head(
        cls, base_model_path: str, path: str, num_labels: int
    ) -> nn.Module:
        model = cls(base_model_path, num_labels)
        model.classifier.load_state_dict(torch.load(path))
        return model

    def forward(
        self, input_ids: torch.tensor, attention_mask: torch.tensor
    ) -> torch.tensor:
        self.base_model.eval()
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids, attention_mask=attention_mask
            )

        B, T, C = outputs.last_hidden_state.shape
        all_tokens_hidden = outputs.last_hidden_state  # (B, T, C)
        last_token_hidden = outputs.last_hidden_state[:, -1, :]  # (B, C)
        last_token_hidden = last_token_hidden.unsqueeze(1).expand(B, T, C)

        combined_representation = torch.cat(
            (all_tokens_hidden, last_token_hidden), dim=-1
        )
        logits = self.classifier(combined_representation)
        return logits


class FineTuneClassifierPhi(nn.Module):
    def __init__(self, base_model_path: str, num_labels: int) -> None:
        super(FineTuneClassifierPhi, self).__init__()
        self.base_model_path = base_model_path
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.base_model.config.hidden_size * 2, num_labels)

    @classmethod
    def from_classifier_head(
        cls, base_model_path: str, path: str, num_labels: int
    ) -> nn.Module:
        model = cls(base_model_path, num_labels)
        model.classifier.load_state_dict(torch.load(path))
        return model

    def forward(
        self, input_ids: torch.tensor, attention_mask: torch.tensor
    ) -> torch.tensor:
        self.base_model.eval()
        with torch.no_grad():
            if "phi-4" in self.base_model_path.lower():
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True,
                    use_cache=False,
                    logits_to_keep=1,
                )
            else:
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True,
                    use_cache=False,
                )

        B, T, C = outputs.hidden_states[-1].shape
        all_tokens_hidden = outputs.hidden_states[-1]  # (B, T, C)
        last_token_hidden = outputs.hidden_states[-1][:, -1, :]  # (B, C)
        last_token_hidden = last_token_hidden.unsqueeze(1).expand(B, T, C)

        combined_representation = torch.cat(
            (all_tokens_hidden, last_token_hidden), dim=-1
        )
        logits = self.classifier(combined_representation)
        return logits


class BaselineClassifier(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        nhead: int,
        max_seq_length: int,
        vocab_size: int,
        pad_token_id: int,
        num_labels: int,
    ) -> None:
        super(BaselineClassifier, self).__init__()
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )

        self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model * 2, num_labels)

    def forward(self, token_ids: torch.tensor) -> torch.tensor:
        batch_size, seq_len = token_ids.shape

        token_emb = self.token_embedding(token_ids)
        pos_ids = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos_ids)
        embeddings = token_emb + pos_emb

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=token_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        output = self.transformer(embeddings, mask=causal_mask)

        B, T, C = output.shape
        all_tokens_hidden = output  # (B, T, C)
        last_token_hidden = output[:, -1, :]  # (B, C)
        last_token_hidden = last_token_hidden.unsqueeze(1).expand(B, T, C)

        combined_representation = torch.cat(
            (all_tokens_hidden, last_token_hidden), dim=-1
        )
        logits = self.classifier(combined_representation)
        return logits
