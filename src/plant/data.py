# src/plant/data.py
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# define dataset class
class TextDataset(Dataset):
    def __init__(
        self,
        encodes_virus,
        encodes_reference=None,  # 参照配列がない場合も考慮
        labels=None,
        censors=None,
        virus=None,
        reference=None,
        dates=None,
        virus_passage=None,
        reference_passage=None,
        #genetic_dist=None,
        weight=None,
    ):
        self.input_ids_virus = encodes_virus["input_ids"]
        self.attention_mask_virus = encodes_virus["attention_mask"]

        # 参照配列がある場合のみ格納

        # labels が None の場合（semantic loss のみ計算する場合）を考慮
        self.input_ids_reference = encodes_reference["input_ids"] if labels is not None else [None] * len(self.input_ids_virus)
        self.attention_mask_reference = encodes_reference["attention_mask"] if labels is not None else [None] * len(self.input_ids_virus)

        self.labels = labels if labels is not None else [None] * len(self.input_ids_virus)
        self.censors = censors if censors is not None else [None] * len(self.input_ids_virus)
        self.virus = virus if virus is not None else [None] * len(self.input_ids_virus)
        self.reference = reference if reference is not None else [None] * len(self.input_ids_virus)
        self.dates = dates if dates is not None else [None] * len(self.input_ids_virus)
        self.virus_passage = virus_passage if virus_passage is not None else [None] * len(self.input_ids_virus)
        self.reference_passage = reference_passage if reference_passage is not None else [None] * len(self.input_ids_virus)
        #self.genetic_dist = genetic_dist if genetic_dist is not None else [None] * len(self.input_ids_virus)
        self.weight = weight if weight is not None else [None] * len(self.input_ids_virus)

    def __getitem__(self, idx):
        item = {
            "input_ids_virus": self.input_ids_virus[idx],
            "attention_mask_virus": self.attention_mask_virus[idx],
        }

        # `None` の場合は適切なデフォルト値を設定
        item["input_ids_reference"] = self.input_ids_reference[idx] if self.input_ids_reference[idx] is not None else torch.zeros_like(self.input_ids_virus[idx])
        item["attention_mask_reference"] = self.attention_mask_reference[idx] if self.attention_mask_reference[idx] is not None else torch.zeros_like(self.attention_mask_virus[idx])

        item["labels"] = torch.tensor(self.labels[idx] if self.labels[idx] is not None else -10.0, dtype=torch.float)
        item["censors"] = torch.tensor(self.censors[idx] if self.censors[idx] is not None else 0.0, dtype=torch.float)
        item["virus"] = torch.tensor(self.virus[idx] if self.virus[idx] is not None else 0, dtype=torch.long)
        item["reference"] = torch.tensor(self.reference[idx] if self.reference[idx] is not None else 0, dtype=torch.long)
        item["dates"] = torch.tensor(self.dates[idx] if self.dates[idx] is not None else 0, dtype=torch.long)
        item["virus_passage"] = torch.tensor(self.virus_passage[idx] if self.virus_passage[idx] is not None else 0, dtype=torch.long)
        item["reference_passage"] = torch.tensor(self.reference_passage[idx] if self.reference_passage[idx] is not None else 0, dtype=torch.long)
        item["weight"] = torch.tensor(self.weight[idx] if self.weight[idx] is not None else 1.0, dtype=torch.float)

        return item

    def __len__(self):
        return len(self.input_ids_virus)


    def get_unique_combinations_indices(self):
        """
        ユニークな組み合わせのインデックスを取得。
        - labels がある場合: (`virus`, `reference`, `virus_passage`, `reference_passage`) に基づく
        - labels がない場合: (`virus`) のみでグループ化
        """
        unique_combinations = {}

        for idx in range(len(self.input_ids_virus)):
            if self.labels[idx] is not None:
                # labels がある場合は (virus, reference, virus_passage, reference_passage) でグループ化
                key = (self.virus[idx], self.reference[idx], self.virus_passage[idx], self.reference_passage[idx])
            else:
                # labels がない場合は virus のみでグループ化
                key = (self.input_ids_virus[idx],)

            if key not in unique_combinations:
                unique_combinations[key] = []
            unique_combinations[key].append(idx)

        return unique_combinations











def tokenize_sequences(seq, tokenizer: PreTrainedTokenizerBase, MAX_LENGTH: int):
    return tokenizer(
        seq, max_length=MAX_LENGTH, padding="max_length",
        truncation=True, return_attention_mask=True, return_tensors="pt",
    )


