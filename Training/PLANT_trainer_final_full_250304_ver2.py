#import library
import argparse
import random
import time
from pathlib import Path
from transformers.utils import ModelOutput
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, BatchSampler, RandomSampler
from itertools import chain, cycle
from Bio import SeqIO
import datetime
from typing import Optional, Tuple
from transformers import (
    AutoModel,
    AutoTokenizer,
    EsmConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    AdamW,
)

start = time.time()

#parse args
parser = argparse.ArgumentParser(
    prog="PLANT trainer",
    description="Run training with season-based-splits of PLANT model",
)

parser.add_argument(
    "--prefix", required=True, help="output prefix"
)
parser.add_argument(
    "--batch_size", default=16, type=int, help="batch size for model training"
)
parser.add_argument(
    "--directory", default=".", help="parent directory for input and output files"
)
parser.add_argument(
    "--checkpoint", required=False, help="directory to resume from checkpoint"
)
parser.add_argument(
    "--num_steps", default=20000, type=int, help="number of steps to train"
)
parser.add_argument(
    "--random-seed", default=42, type=int, help="random seed for reproducibility"
)
parser.add_argument(
    "--learning-rate", default=1e-4, type=float, help="learning rate for training"
)
parser.add_argument(
    "--max-saves", default=1, type=int, help="max number of checkpoints to save"
)

parser.add_argument("--CSE_w", default=0, type=float, help="Weight for CSE")
parser.add_argument("--CSE_w_virus_only", default=0, type=float, help="Weight for CSE virus only")

parser.add_argument("--semantic_w", default=0.2, type=float, help="Weight for semantic loss")
parser.add_argument("--semantic_w_virus_only", default=0.2, type=float, help="Weight for semantic loss virus only")
parser.add_argument("--cart_w", default=0.05, type=float, help="Weight for cart loss")


parser.add_argument("--dropout_regressor", default=0.05, type=float, help="Dropout rate for regressor")
parser.add_argument("--reg_intermediate_dim", default=256, type=int, help="Intermediate_dim for regressor")

parser.add_argument("--CSE_alpha", default=0.0, type=float, help="CSE alpha")


parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decayin AdamW")
parser.add_argument("--reg_weight_decay", default=m, type=float, help="Weight decayin AdamW for regressor")

parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D", help="model path")

parser.add_argument("--intermediate_dim_encoder", default=64, type=int, help="intermediate dim encoder")
parser.add_argument("--dropout_encoder", default=0.1, type=float,  help="dropout encoder")
parser.add_argument("--lg_w", default=0.01, type=float,  help="Local grobal W")

args = parser.parse_args()

VERSION = args.prefix

storage_path = args.directory  # Dot for project directory
outputs_path = f"{storage_path}/Season_based_split_performance/{VERSION}/trained_until_full/"
Path(outputs_path).mkdir(parents=True, exist_ok=True)

MODEL_NAME = args.model

MAX_LENGTH = 329
NUM_STEPS = args.num_steps
BATCH_SIZE = args.batch_size
RANDSEED = args.random_seed
LEARNING_RATE = args.learning_rate
CHECKPOINT = args.checkpoint
SAVE_TOTAL_LIMIT = args.max_saves
CSE_W = args.CSE_w
SEMANTIC_W = args.semantic_w

DROPOUT_REGRESSOR = args.dropout_regressor
REG_INTERMEDIATE_DIM = args.reg_intermediate_dim
CSE_ALPHA = args.CSE_alpha

WEIGHT_DECAY = args.weight_decay
REG_WEIGHT_DECAY = args.reg_weight_decay

CSE_W_VIRUS_ONLY = args.CSE_w_virus_only
SEMANTIC_W_VIRUS_ONLY= args.semantic_w_virus_only

CART_W=args.cart_w

DROPOUT_ENCODER=args.dropout_encoder
INTERMEDIATE_DIM_ENCODER=args.intermediate_dim_encoder
LG_W=args.lg_w

dataset_stem_dir_path = (
    f"{storage_path}/"
)


#read data

#dataset for unsupervised learning
HA_fas_name = storage_path + "/data/ncbiflu_HA_all_110424_noX_clu99_aln_realign2.fas"
HA_metadata_file = storage_path + "/data/ncbiflu_HA_all_110424_noX_clu99_meta.csv"


def fasta_to_dataframe(fasta_file):
    ids = []
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        ids.append(record.id)
        sequences.append(str(record.seq))

    df = pd.DataFrame({
        "ID": ids,
        "Sequence": sequences
    })
    return df

#DA_df: dataset for unsupervised learning (sequence)
DA_df = fasta_to_dataframe(HA_fas_name)

DA_df = DA_df.rename(columns = {'Sequence':'seq'})
DA_df['seq'] = DA_df['seq'].str.replace("-", "")


DA_metadata_df = pd.read_csv(HA_metadata_file)
DA_metadata_df = DA_metadata_df.rename(columns = {'name':'ID'})
DA_df = pd.merge(DA_df, DA_metadata_df, on='ID', how='inner')

DA_df['year'] = DA_df['date'].str.slice(0, 4)
DA_df['HA_type'] = DA_df['serotype'].str.slice(0, 2)

DA_df = DA_df[DA_df['year'].str.isnumeric()]

DA_df['year'] = DA_df['year'].astype(int)
DA_df = DA_df.reset_index(drop=True)

DA_df = DA_df[DA_df['seq'].str.count("X") == 0]


# complete date
def complete_date(date_str):
    try:
        # return the date as is if it is already complete
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        pass

    try:
        # If format is YYYY-MM-, fill with the last day of the month
        if date_str.endswith("-"):
            year, month = date_str.split("-")[:2]
            last_day = str((datetime.datetime.strptime(f"{year}-{int(month)+1}-01", "%Y-%m-%d") - datetime.timedelta(days=1)).day)
            return f"{year}-{month}-{last_day}"
    except ValueError:
        pass

    try:
        # If format is YYYY--, fill with December 31
        if date_str.count("-") == 2:
            year = date_str.split("-")[0]
            return f"{year}-12-31"
    except ValueError:
        pass

    return date_str  # if none of the above applies, return as is


DA_df["date_corrected"] = DA_df["date"].apply(complete_date)
DA_df["date_corrected"] = pd.to_datetime(DA_df["date_corrected"], format="%Y-%m-%d", errors="coerce")



# input dataset
print("Now loading input dataset")
print(f"Model used: {MODEL_NAME}")

print(f"Start performing antigenicity prediction ...")

data_path = (
    dataset_stem_dir_path
    + f"data/WHO_GISAID_dataset_final_strict_score_250124.csv"
)

WHO_GISAID_dataset_df = pd.read_csv(data_path)

#Apply filters sequentially
WHO_GISAID_dataset_df = WHO_GISAID_dataset_df[
    ~WHO_GISAID_dataset_df["virus_seq"].str.contains("X")
    & ~WHO_GISAID_dataset_df["reference_seq"].str.contains("X")
    & ~WHO_GISAID_dataset_df["virus_seq"].str.contains("B")
    & ~WHO_GISAID_dataset_df["virus_seq"].str.contains(r"\*")
    & ~WHO_GISAID_dataset_df["reference_seq"].str.contains(r"\*")
    # & ~WHO_GISAID_dataset_df["pair_strict"].str.contains("UNKNOWN")
    & ~WHO_GISAID_dataset_df["pair_strict"].str.contains(r"\|")
]

WHO_GISAID_dataset_df = WHO_GISAID_dataset_df[
    WHO_GISAID_dataset_df["virus_seq"].str.len() == 329
]
WHO_GISAID_dataset_df = WHO_GISAID_dataset_df[
    WHO_GISAID_dataset_df["reference_seq"].str.len() == 329
]

# remove blacklist
# 1. Extract rows where virus and reference are identical and score >= 0.25, create blacklist_df
blacklist_df = WHO_GISAID_dataset_df[
    (WHO_GISAID_dataset_df['virus'] == WHO_GISAID_dataset_df['reference']) &
    (WHO_GISAID_dataset_df['score'] >= 0.25)
][['date', 'reference', 'reference_passage']].drop_duplicates()

# 2. Remove combinations (date, reference, reference_passage) in blacklist_df, create WHO_GISAID_dataset_df_filtered
WHO_GISAID_dataset_df_filtered = WHO_GISAID_dataset_df.merge(
    blacklist_df, on=['date', 'reference', 'reference_passage'], how='left', indicator=True
).query('_merge == "left_only"').drop(columns=['_merge'])

# 3. Extract only combinations in blacklist_df, create WHO_GISAID_dataset_df_deleted
WHO_GISAID_dataset_df_deleted = WHO_GISAID_dataset_df.merge(
    blacklist_df, on=['date', 'reference', 'reference_passage'], how='inner'
)

print(WHO_GISAID_dataset_df.shape)
print(WHO_GISAID_dataset_df_filtered.shape)


# clean dataset
selected_df = WHO_GISAID_dataset_df_filtered[
    [
        "date",
        "virus",
        "reference",
        "virus_passage",
        "reference_passage",
        "score",
        "censor",
        "virus_seq",
        "reference_seq",
        "virus_collection_date",
        "reference_collection_date",
    ]
]
selected_df = selected_df.dropna()
selected_df = selected_df.sample(frac=1, random_state=42)
selected_df.reset_index(drop=True, inplace=True)

selected_df['virus_collection_year'] = pd.to_datetime(selected_df['virus_collection_date']).dt.year


# define categorical variables to account for batch effects
selected_df["date_category"] = selected_df["date"].astype("category").cat.codes
num_date_categories = selected_df["date_category"].nunique()

selected_df["virus_category"] = selected_df["virus"].astype("category").cat.codes
num_virus_categories = selected_df["virus_category"].nunique()

selected_df["reference_category"] = (
    selected_df["reference"].astype("category").cat.codes
)
num_reference_categories = selected_df["reference_category"].nunique()

selected_df["virus_passage_category"] = (
    selected_df["virus_passage"].astype("category").cat.codes
)
num_virus_passage_categories = selected_df["virus_passage_category"].nunique()

selected_df["reference_passage_category"] = (
    selected_df["virus_passage"].astype("category").cat.codes
)
num_reference_passage_categories = selected_df["reference_passage_category"].nunique()

print("num_date_categories", num_date_categories)
print("num_virus_categories", num_virus_categories)
print("num_reference_categories", num_reference_categories)
print("num_virus_passage_categories", num_virus_passage_categories)
print("num_reference_passage_categories", num_reference_passage_categories)


# Function to split the dataframe using StratifiedGroupKFold
def split_dataframe_with_stratified_group_kfold(df, train_ratio=0.8, seed=RANDSEED):
    # Prepare for stratified group k-fold
    skf = StratifiedGroupKFold(n_splits=int(1 / (1 - train_ratio)), shuffle=True, random_state=seed)

    # Extract groups and stratification labels
    groups = df["virus"]  # Group by virus
    stratify_labels = df["virus_collection_year"]  # Stratify by year

    # Perform the split
    for train_idx, val_idx in skf.split(df, stratify_labels, groups):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        break  # Only take the first split as we only need train/validation

    return train_df, val_df

# separate training and test data based on the dataset_type column
selected_df_train_val, selected_df_test = split_dataframe_with_stratified_group_kfold(selected_df, train_ratio=0.8)


#data weight according to data density
#set weight 1 or test dataset
selected_df_test.loc[:, "weight"] = 1

# 1. Divide the score range (0-1) into 9 bins
num_bins = 9
bin_edges = np.linspace(0, 1, num_bins + 1)

# 2. Group by unique combinations of virus, reference, virus_passage, and reference_passage, and compute the group mean
unique_group_means = (
    selected_df_train_val
    .groupby(['virus', 'reference', 'virus_passage', 'reference_passage'])
    ['score']
    .mean()
    .reset_index()
)

# 3. Assign bins based on group mean values
unique_group_means['bin'] = pd.cut(
    unique_group_means['score'], bins=bin_edges, labels=False, include_lowest=True
)

# 4. Calculate bin frequencies
bin_counts = unique_group_means['bin'].value_counts().sort_index()

# 5. Calculate weights based on frequencies (rarer bins get larger weights)
total_samples = len(unique_group_means)
bin_weights = {bin_idx: total_samples / count for bin_idx, count in bin_counts.items()}

# 6. Assign weights to each data point
unique_group_means['weight'] = unique_group_means['bin'].map(bin_weights)

# Merge results into selected_df_train_val
selected_df_train_val = selected_df_train_val.merge(
    unique_group_means[['virus', 'reference', 'virus_passage', 'reference_passage', 'bin', 'weight']],
    on=['virus', 'reference', 'virus_passage', 'reference_passage'],
    how='left'
)

# 1. Get the minimum weight
min_weight = selected_df_train_val['weight'].min()

# 2. Normalize so that the minimum weight is fixed to 1
selected_df_train_val['weight'] = selected_df_train_val['weight'] / min_weight

# 3. Clip weights at an upper limit of 5
selected_df_train_val['weight'] = selected_df_train_val['weight'].clip(upper=5)

bin_weight_mean = selected_df_train_val.groupby('bin')['weight'].mean()
print(bin_weight_mean)
print(unique_group_means.shape)


selected_df_train, selected_df_val = split_dataframe_with_stratified_group_kfold(selected_df_train_val, train_ratio=8/9)


print("Training data size: ", len(selected_df_train))
print("Validation data size: ", len(selected_df_val))
print("Test data size: ", len(selected_df_test))



# Define tokenizer, encode sequences, and create dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_sequences(seq, tokenizer, MAX_LENGTH):
    return tokenizer(
        seq,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )


encodes_virus_train = tokenize_sequences(
    selected_df_train["virus_seq"].tolist(), tokenizer, MAX_LENGTH
)
encodes_virus_val = tokenize_sequences(
    selected_df_val["virus_seq"].tolist(), tokenizer, MAX_LENGTH
)
encodes_virus_test = tokenize_sequences(
    selected_df_test["virus_seq"].tolist(), tokenizer, MAX_LENGTH
)

encodes_reference_train = tokenize_sequences(
    selected_df_train["reference_seq"].tolist(), tokenizer, MAX_LENGTH
)
encodes_reference_val = tokenize_sequences(
    selected_df_val["reference_seq"].tolist(), tokenizer, MAX_LENGTH
)
encodes_reference_test = tokenize_sequences(
    selected_df_test["reference_seq"].tolist(), tokenizer, MAX_LENGTH
)


encodes_virus_all_HA = tokenize_sequences(
    DA_df["seq"].tolist(), tokenizer, MAX_LENGTH
)

print("Dataset encoding completed")



# define dataset class
class TextDataset(Dataset):
    def __init__(
        self,
        encodes_virus,
        encodes_reference=None,  # consider the case where no reference sequence is provided
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

        # store only if reference sequence is provided

        # consider the case where labels is None (when calculating only semantic loss)
        self.input_ids_reference = encodes_reference["input_ids"] if labels is not None else [None] * len(self.input_ids_virus)
        self.attention_mask_reference = encodes_reference["attention_mask"] if labels is not None else [None] * len(self.input_ids_virus)

        self.labels = labels if labels is not None else [None] * len(self.input_ids_virus)
        self.censors = censors if censors is not None else [None] * len(self.input_ids_virus)
        self.virus = virus if virus is not None else [None] * len(self.input_ids_virus)
        self.reference = reference if reference is not None else [None] * len(self.input_ids_virus)
        self.dates = dates if dates is not None else [None] * len(self.input_ids_virus)
        self.virus_passage = virus_passage if virus_passage is not None else [None] * len(self.input_ids_virus)
        self.reference_passage = reference_passage if reference_passage is not None else [None] * len(self.input_ids_virus)
        self.weight = weight if weight is not None else [None] * len(self.input_ids_virus)

    def __getitem__(self, idx):
        item = {
            "input_ids_virus": self.input_ids_virus[idx],
            "attention_mask_virus": self.attention_mask_virus[idx],
        }

        # if `None`, set appropriate default values
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
        Get indices of unique combinations.
        - If labels are available: group by (virus, reference, virus_passage, reference_passage)
        - If labels are not available: group only by (virus)
        """
        unique_combinations = {}

        for idx in range(len(self.input_ids_virus)):
            if self.labels[idx] is not None:
                # if labels are available, group by (virus, reference, virus_passage, reference_passage)
                key = (self.virus[idx], self.reference[idx], self.virus_passage[idx], self.reference_passage[idx])
            else:
                # if labels are not available, group only by virus
                key = (self.input_ids_virus[idx],)

            if key not in unique_combinations:
                unique_combinations[key] = []
            unique_combinations[key].append(idx)

        return unique_combinations


# define dataset class for only embedding distance calculation
class TextDataset_only_embedding(Dataset):
    def __init__(self, encodes_virus):
        self.input_ids_virus = encodes_virus["input_ids"]

    def __getitem__(self, idx):
        item = {
            "input_ids_virus": self.input_ids_virus[idx],
        }
        return item

    def __len__(self):
        return len(self.input_ids_virus)


print(selected_df_train)


# %%
# define categorical variables
date_categories = torch.tensor(
    selected_df_train["date_category"].values, dtype=torch.long
)
virus_passage_categories = torch.tensor(
    selected_df_train["virus_passage_category"].values, dtype=torch.long
)
reference_passage_categories = torch.tensor(
    selected_df_train["reference_passage_category"].values, dtype=torch.long
)

# load dataset
dataset_train = TextDataset(
    encodes_virus_train,
    encodes_reference_train,
    selected_df_train["score"].tolist(),
    selected_df_train["censor"].tolist(),
    selected_df_train["virus_category"].tolist(),
    selected_df_train["reference_category"].tolist(),
    selected_df_train["date_category"].tolist(),
    selected_df_train["virus_passage_category"].tolist(),
    selected_df_train["reference_passage_category"].tolist(),
    selected_df_train["weight"].tolist(),
)
dataset_val = TextDataset(
    encodes_virus_val,
    encodes_reference_val,
    selected_df_val["score"].tolist(),
    selected_df_val["censor"].tolist(),
    selected_df_val["virus_category"].tolist(),
    selected_df_val["reference_category"].tolist(),
    selected_df_val["date_category"].tolist(),
    selected_df_val["virus_passage_category"].tolist(),
    selected_df_val["reference_passage_category"].tolist(),
    selected_df_val["weight"].tolist(),
)
dataset_test = TextDataset(
    encodes_virus_test,
    encodes_reference_test,
    selected_df_test["score"].tolist(),
    selected_df_test["censor"].tolist(),
    selected_df_test["virus_category"].tolist(),
    selected_df_test["reference_category"].tolist(),
    selected_df_test["date_category"].tolist(),
    selected_df_test["virus_passage_category"].tolist(),
    selected_df_test["reference_passage_category"].tolist(),
    selected_df_test["weight"].tolist(),
)


print(f"Training data size: {len(dataset_train)}")
print(f"Validation data size: {len(dataset_val)}")
print(f"Test data size: {len(dataset_test)}")

# %%
# define model class for embedding distance calculation only
class semanticESM_embedding_dist_only(PreTrainedModel):
    config_class = EsmConfig

    def __init__(self, config, esm_model_name):
        super(semanticESM_embedding_dist_only, self).__init__(config)
        self.esm_model = AutoModel.from_pretrained(
            esm_model_name, add_pooling_layer=False
        )
        self.embedding_dim = self.esm_model.config.hidden_size
        # freeze model parameters

    def forward(self, input_ids_virus, attention_mask_virus, input_ids_reference, attention_mask_reference, **kwargs):
        # output from the ESM model
        virus_encoder_out = self.esm_model(
            input_ids_virus, attention_mask=attention_mask_virus
        ).last_hidden_state  # (batch_size, seq_len, hidden_dim)

        reference_encoder_out = self.esm_model(
            input_ids_reference, attention_mask=attention_mask_reference
        ).last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # Mean Pooling (considering mask)
        masked_sum_virus = (virus_encoder_out * attention_mask_virus.unsqueeze(-1)).sum(dim=1)
        mask_count_virus = attention_mask_virus.sum(dim=1, keepdim=True).clamp(min=1)
        mean_pooled_virus = masked_sum_virus / mask_count_virus  # (batch_size, hidden_dim)

        masked_sum_reference = (reference_encoder_out * attention_mask_reference.unsqueeze(-1)).sum(dim=1)
        mask_count_reference = attention_mask_reference.sum(dim=1, keepdim=True).clamp(min=1)
        mean_pooled_reference = masked_sum_reference / mask_count_reference  # (batch_size, hidden_dim)

        # Max Pooling
        max_pooled_virus = torch.max(virus_encoder_out, dim=1)[0]  # (batch_size, hidden_dim)
        max_pooled_reference = torch.max(reference_encoder_out, dim=1)[0]  # (batch_size, hidden_dim)

        # concatenate Mean + Max
        virus_embedding = torch.cat([mean_pooled_virus, max_pooled_virus], dim=-1)  # (batch_size, hidden_dim * 2)
        reference_embedding = torch.cat([mean_pooled_reference, max_pooled_reference], dim=-1)  # (batch_size, hidden_dim * 2)

        # compute Euclidean distance
        distance = torch.norm(virus_embedding - reference_embedding, p=2, dim=1, keepdim=True)

        return ModelOutput(logits=distance)

esm_config = EsmConfig.from_pretrained(MODEL_NAME)
embed_model = semanticESM_embedding_dist_only(esm_config, MODEL_NAME)


embed_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_model.to(device).half()


batch_size = 128
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

embed_dist_train = []

# perform prediction batch by batch, train
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
    for batch in dataloader_train:
        input_ids_virus = batch['input_ids_virus'].to(device)
        input_ids_reference = batch['input_ids_reference'].to(device)
        attention_mask_virus = batch['attention_mask_virus'].to(device)
        attention_mask_reference = batch['attention_mask_reference'].to(device)

        outputs = embed_model(input_ids_virus, attention_mask_virus, input_ids_reference, attention_mask_reference)

        embed_dist_train.append(outputs.logits.cpu().numpy())

embed_dist_train = np.concatenate(embed_dist_train, axis=0)

selected_df_train["embed_dist"] = embed_dist_train.flatten().tolist()

embed_scale_factor = selected_df_train['embed_dist'].quantile(0.99)
print(f"embed_scale_factor: {embed_scale_factor}")



dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

embed_dist_test = []

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
    for batch in dataloader_test:
        input_ids_virus = batch['input_ids_virus'].to(device)
        input_ids_reference = batch['input_ids_reference'].to(device)
        attention_mask_virus = batch['attention_mask_virus'].to(device)
        attention_mask_reference = batch['attention_mask_reference'].to(device)

        outputs = embed_model(input_ids_virus, attention_mask_virus, input_ids_reference, attention_mask_reference)

        embed_dist_test.append(outputs.logits.cpu().numpy())

embed_dist_test = np.concatenate(embed_dist_test, axis=0)

selected_df_test["embed_dist"] = embed_dist_test.flatten().tolist()



#dataset combine
dataset_virus_only = TextDataset(
    encodes_virus_all_HA
)

dataset_train_combined_pretrain = ConcatDataset([dataset_virus_only]) #ConcatDataset([dataset_train, dataset_virus_only])
dataset_train_combined = ConcatDataset([dataset_train, dataset_virus_only])


# %%
############################################## Train the model ##############################################
# One Hot Encode effects for use in systematic error

OHE_virus = OneHotEncoder(handle_unknown="ignore").fit(
    [[x] for x in selected_df_train["virus_category"]]
)
OHE_ref = OneHotEncoder(handle_unknown="ignore").fit(
    [[x] for x in selected_df_train["reference_category"]]
)
OHE_date = OneHotEncoder(handle_unknown="ignore").fit(
    [[x] for x in selected_df_train["date_category"]]
)
OHE_vp = OneHotEncoder(handle_unknown="ignore").fit(
    [[x] for x in selected_df_train["virus_passage_category"]]
)
OHE_rp = OneHotEncoder(handle_unknown="ignore").fit(
    [[x] for x in selected_df_train["reference_passage_category"]]
)

effects_len = (
    #len(OHE_virus.categories_[0])
    len(OHE_ref.categories_[0])
    + len(OHE_vp.categories_[0])
    + len(OHE_rp.categories_[0])
)

virus_effects_len = len(OHE_virus.categories_[0])


# TODO: Move this to after model training?
joblib.dump(OHE_virus, f"{outputs_path}/virus_encoder.joblib")
joblib.dump(OHE_ref, f"{outputs_path}/ref_encoder.joblib")
joblib.dump(OHE_date, f"{outputs_path}/date_encoder.joblib")
joblib.dump(OHE_vp, f"{outputs_path}/vp_encoder.joblib")
joblib.dump(OHE_rp, f"{outputs_path}/rp_encoder.joblib")


# PLANT model ######################################################
class semanticESM(PreTrainedModel):
    config_class = EsmConfig

    def __init__(
        self,
        config,
        esm_model_name,
        effects_len = effects_len,
        virus_effects_len = virus_effects_len,
        embed_scale_factor = 1,
        latent_dim=3,
        intermediate_dim = 256,
        intermediate_dim_encoder = 64,
        dropout = 0.05,
        dropout_encoder=0.1,
        MAIN_W=1,
        CSE_W=0.001,
        CSE_ALPHA=0,
        SEMANTIC_W=0.1,
        CSE_W_VIRUS_ONLY=0.02,
        SEMANTIC_W_VIRUS_ONLY=0.2,
        CART_W=0.05,
        LG_W=0.05
    ):
        super().__init__(config)

        # initialize ESM model
        self.esm_model = AutoModel.from_pretrained(esm_model_name, add_pooling_layer=False)
        self.esm_model_original = self._initialize_frozen_esm_model(esm_model_name)

        self.embedding_dim = self.esm_model.config.hidden_size

        # regression model
        self.regressor = nn.Sequential(
            nn.Linear(self.embedding_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, latent_dim),
        )

        # systematic error
        self.virus_effects = nn.Linear(virus_effects_len, 1, bias=False)
        self.systematic_error_effects = nn.Sequential(
            nn.Linear(effects_len, intermediate_dim_encoder),
            nn.ReLU(),
            nn.Dropout(dropout_encoder),
            nn.Linear(intermediate_dim_encoder, 1, bias=False),
        )

        # scale parameter
        self.embed_scale = nn.Parameter(torch.tensor(1.0))

        # MSE Loss
        self.mse_loss = nn.MSELoss()
        self.mse_loss_wo_mean = nn.MSELoss(reduction="none")


        # save hyperparameters
        self.MAIN_W = MAIN_W
        self.CSE_W = CSE_W
        self.SEMANTIC_W = SEMANTIC_W
        self.CSE_ALPHA=CSE_ALPHA
        self.CART_W = CART_W
        self.CSE_W_VIRUS_ONLY = CSE_W_VIRUS_ONLY
        self.SEMANTIC_W_VIRUS_ONLY = SEMANTIC_W_VIRUS_ONLY
        self.LG_W = LG_W

        self.embed_scale_factor = embed_scale_factor

    def _initialize_frozen_esm_model(self, esm_model_name):
        """Load pretrained ESM model and freeze its parameters"""
        esm_model = AutoModel.from_pretrained(esm_model_name, add_pooling_layer=False)
        for param in esm_model.parameters():
            param.requires_grad = False
        return esm_model

    def encode_sequence(self, model, input_ids, attention_mask):
        """Encode sequence with ESM model and extract features"""
        encoder_out = model(input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
        return encoder_out.last_hidden_state[:, 0, :]

    def encode_one_hot(self, encoder, input_tensor):
        """One-hot encoding を適用"""
        model_dtype = next(self.parameters()).dtype
        return torch.tensor(
            encoder.transform(input_tensor.cpu().numpy().reshape(-1, 1)).toarray(),
            dtype=model_dtype,
        ).to(self.device)

    def extract_pooled_embeddings(self, model, input_ids, attention_mask):
        """Feature extraction with original ESM (Mean Pooling + Max Pooling)"""
        with torch.no_grad():
            encoder_out = model(input_ids, attention_mask).last_hidden_state

        masked_sum = (encoder_out * attention_mask.unsqueeze(-1)).sum(dim=1)
        mask_count = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_pooled = masked_sum / mask_count

        max_pooled = torch.max(encoder_out, dim=1)[0]
        pooled_embedding = torch.cat([mean_pooled, max_pooled], dim=-1)

        return pooled_embedding


    def compute_semantic_loss(self, latents, latents_original, embed_scale, embed_scale_factor):

        # calculate pairwise distances between embeddings
        pairwise_distances = torch.cdist(latents, latents, p=2)
        upper_triangle_indices = torch.triu_indices(
            pairwise_distances.size(0), pairwise_distances.size(1), offset=1
        )

        pairwise_distances_original = torch.cdist(
            latents_original,
            latents_original,
            p=2
        )
        pairwise_distances_original = pairwise_distances_original / embed_scale_factor

        # calculate MSE Loss (apply weights)
        semantic_loss = self.mse_loss_wo_mean(
            pairwise_distances_original[upper_triangle_indices],
            pairwise_distances[upper_triangle_indices] * embed_scale
        ).mean()

        return semantic_loss


    def contrastive_loss_semantic(self, embeddings1, embeddings2, margin=1.0, alpha=1):
        """
        Improved Contrastive Loss Semantic1 to penalize closer negatives more heavily
        - `alpha`: penalty strength for close negatives (larger = stronger Hard Negative emphasis)
        Note: The final version does not use this contrastive_loss_semantic
        """
        batch_size = embeddings1.shape[0]

        # calculate L2 distance
        distance_matrix = torch.cdist(embeddings1, embeddings2, p=2)  # shape: (batch_size, batch_size)

        # minimize positive pairs (diagonal elements)
        positive_loss = torch.mean(torch.diag(distance_matrix))  # minimize D[i, i]

        # design so that closer negatives are penalized more strongly
        # (1) apply usual margin-based clamping
        margin_loss = torch.clamp(margin - distance_matrix, min=0)  # enforce D[i, j] (i≠j) ≥ margin

        # (2) apply exp(-α * distance) so closer pairs have larger weight
        weight = torch.exp(-alpha * distance_matrix)  # closer pairs get larger weights
        negative_loss = torch.mean(weight * margin_loss)  # weighted penalty

        return positive_loss + negative_loss

    def local_global_loss(self, latents, k_local=3, margin_global=0.125):
        """
        - attract k_local nearest neighbors (preserve local density)
        - repel distant pairs not farther than margin_global (encourage cluster separability)
        """
        # distance matrix: shape (N, N)
        dists = torch.cdist(latents, latents, p=2)
        N = latents.size(0)

        # exclude diagonal elements (self)
        eye_mask = torch.eye(N, device=dists.device).bool()
        dists_no_self = dists.masked_fill(eye_mask, float('inf'))

        ### 1. Local attraction (bring neighbors closer)
        # restrict k_local to N-1
        k_safe = min(k_local, N - 1)

        if k_safe > 0:
            knn_dists, _ = torch.topk(dists_no_self, k=k_safe, largest=False, dim=1)
            local_loss = torch.mean(knn_dists)
        else:
            local_loss = torch.tensor(0.0, device=latents.device)

        ### 2. Global repulsion (separate distant clusters)
        margin_mask = dists_no_self < margin_global
        repel_loss = torch.clamp(margin_global - dists_no_self, min=0.0)
        global_loss = torch.sum(repel_loss * margin_mask) / (margin_mask.sum() + 1e-8)  # avoid div by 0

        return local_loss + global_loss


    def custom_loss(
        self,
        predictions,
        predictions_cart,
        targets,
        censors,
        virus_regressor_out,
        virus_regressor_out2,
        reference_regressor_out,
        reference_regressor_out2,
        virus_embedding_original,
        reference_embedding_original,
        weight
    ):

        uncensored_loss = self.mse_loss_wo_mean(predictions, targets) * (1 - censors)
        censored_loss = self.mse_loss_wo_mean(predictions, torch.minimum(predictions, targets)) * censors

        uncensored_loss = uncensored_loss * weight
        censored_loss = censored_loss * weight

        uncensored_loss_cart = self.mse_loss_wo_mean(predictions_cart, targets) * (1 - censors)
        censored_loss_cart = self.mse_loss_wo_mean(predictions_cart, torch.minimum(predictions_cart, targets)) * censors

        uncensored_loss_cart = uncensored_loss_cart * weight
        censored_loss_cart = censored_loss_cart * weight

        combined_latents = torch.cat([virus_regressor_out, reference_regressor_out], dim=0)
        combined_latents2 = torch.cat([virus_regressor_out2, reference_regressor_out2], dim=0)
        combined_latents_original = torch.cat([virus_embedding_original, reference_embedding_original], dim=0)

        # **Semantic CSE: "repel others" Contrastive Loss**
        contrastive_loss_value = self.contrastive_loss_semantic(combined_latents, combined_latents2, alpha=self.CSE_ALPHA)
        local_global_loss_value = self.local_global_loss(combined_latents, k_local=3, margin_global=0.125)


        semantic_loss = self.compute_semantic_loss(combined_latents,combined_latents_original,self.embed_scale,self.embed_scale_factor)

        total_loss = (
            torch.mean(uncensored_loss) * self.MAIN_W
            + torch.mean(censored_loss) * self.MAIN_W
            + torch.mean(uncensored_loss_cart) * self.CART_W
            + torch.mean(censored_loss_cart) * self.CART_W
            + contrastive_loss_value * self.CSE_W
            + semantic_loss * self.SEMANTIC_W
            + local_global_loss_value * self.LG_W
        )

        return total_loss



    def custom_loss_only_semantic(
        self,
        virus_regressor_out,
        virus_regressor_out2,
        virus_embedding_original
    ):
        """
        - unsupervised learing for data without label (i.e., antigenic distance)
        """
        contrastive_loss_value = self.contrastive_loss_semantic(virus_regressor_out, virus_regressor_out2, alpha=self.CSE_ALPHA)
        local_global_loss_value = self.local_global_loss(virus_regressor_out, k_local=3, margin_global=0.125)

        semantic_loss = self.compute_semantic_loss(virus_regressor_out,virus_embedding_original,self.embed_scale,self.embed_scale_factor)

        total_loss = (
            semantic_loss * self.SEMANTIC_W_VIRUS_ONLY
            + contrastive_loss_value * self.CSE_W_VIRUS_ONLY
            + local_global_loss_value * self.LG_W
        )
        return total_loss

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        **kwargs,
    ):
        from transformers import EsmConfig
        from safetensors.torch import load_file
        from pathlib import Path

        if config is None:
            config = EsmConfig.from_pretrained(pretrained_model_name_or_path)

        model = cls(config, *model_args, **kwargs)

        checkpoint_dir = Path(pretrained_model_name_or_path)
        safetensor_files = sorted(checkpoint_dir.glob("model-*-of-*.safetensors"))

        print(f"[INFO] Found {len(safetensor_files)} safetensors.")

        state_dict = {}
        for f in safetensor_files:
            print(f"[INFO] Loading {f}")
            part = load_file(str(f), device="cpu")
            state_dict.update(part)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("[INFO] Model state_dict loaded.")
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

        return model

    def forward(
        self,
        input_ids_virus: torch.Tensor,
        attention_mask_virus: torch.Tensor,
        input_ids_reference: Optional[torch.Tensor] = None,
        attention_mask_reference: Optional[torch.Tensor] = None,
        censors=None,
        labels=None,
        virus=None,
        reference=None,
        dates=None,
        virus_passage=None,
        reference_passage=None,
        weight=None,
        **kwargs,
    ):
        
        
        # check if labels are None for each sample
        has_labels_mask = labels.ne(float(-10.0)) if labels is not None else torch.zeros(input_ids_virus.shape[0], dtype=torch.bool, device=self.device)
        # handle case where `input_ids_reference` is None (avoid DataLoader KeyError)
        if input_ids_reference is None:
            input_ids_reference = torch.zeros_like(input_ids_virus, device=self.device)
            attention_mask_reference = torch.zeros_like(attention_mask_virus, device=self.device)
        
        # split data based on whether labels exist
        virus_regressor_out = self.regressor(self.encode_sequence(self.esm_model, input_ids_virus, attention_mask_virus))
        virus_regressor_out2 = self.regressor(self.encode_sequence(self.esm_model, input_ids_virus, attention_mask_virus))

        if self.training == False and labels is None:
            return ModelOutput(hidden_state_virus=virus_regressor_out)

        virus_embedding_original = self.extract_pooled_embeddings(self.esm_model_original, input_ids_virus, attention_mask_virus)

        # for samples without labels, calculate custom_loss_only_semantic
        virus_regressor_out_no_labels = virus_regressor_out[~has_labels_mask]
        virus_regressor_out2_no_labels = virus_regressor_out2[~has_labels_mask]
        virus_embedding_original_no_labels = virus_embedding_original[~has_labels_mask]
        
        if virus_regressor_out_no_labels.shape[0] > 0:
            loss_no_labels = self.custom_loss_only_semantic(virus_regressor_out_no_labels, virus_regressor_out2_no_labels, virus_embedding_original_no_labels)
        else:
            loss_no_labels = torch.tensor(0.0, device=self.device)
        
        # for samples with labels, process reference
        if has_labels_mask.any():
            input_ids_reference = input_ids_reference[has_labels_mask]
            attention_mask_reference = attention_mask_reference[has_labels_mask]
            reference_regressor_out = self.regressor(self.encode_sequence(self.esm_model, input_ids_reference, attention_mask_reference))
            reference_regressor_out2 = self.regressor(self.encode_sequence(self.esm_model, input_ids_reference, attention_mask_reference))
            
            distance = torch.norm(virus_regressor_out[has_labels_mask] - reference_regressor_out, p=2, dim=1, keepdim=True)
            
            virus_encoding = self.encode_one_hot(OHE_virus, virus[has_labels_mask])
            reference_encoding = self.encode_one_hot(OHE_ref, reference[has_labels_mask])
            virus_passage_encoding = self.encode_one_hot(OHE_vp, virus_passage[has_labels_mask])
            reference_passage_encoding = self.encode_one_hot(OHE_rp, reference_passage[has_labels_mask])
            
            combined_encoding = torch.cat([reference_encoding, virus_passage_encoding, reference_passage_encoding], dim=-1)
            systematic_error1 = self.virus_effects(virus_encoding)
            systematic_error2 = self.systematic_error_effects(combined_encoding)
            systematic_error = systematic_error1 + systematic_error2
            observed_distance = distance + systematic_error
            
            logits = torch.cat((observed_distance, distance), dim=1)
            reference_embedding_original = self.extract_pooled_embeddings(self.esm_model_original, input_ids_reference, attention_mask_reference)
            
            combined_loss = self.custom_loss(
                observed_distance,
                distance,
                labels[has_labels_mask].to(self.device).view(-1, 1),
                censors[has_labels_mask].to(self.device).view(-1, 1),
                virus_regressor_out[has_labels_mask],
                virus_regressor_out2[has_labels_mask],
                reference_regressor_out,
                reference_regressor_out2,
                virus_embedding_original[has_labels_mask],
                reference_embedding_original,
                weight[has_labels_mask].to(self.device).view(-1, 1)
            )
            systematic_error1_L2 = torch.mean(systematic_error1 ** 2)
            systematic_error2_L2 = torch.mean(systematic_error2 ** 2)
            combined_loss = combined_loss + (systematic_error1_L2 + systematic_error2_L2) * 1.0E-4
        else:
            combined_loss = torch.tensor(0.0, device=self.device)
            logits = None
        
        total_loss = combined_loss + loss_no_labels
        
        return ModelOutput(loss=total_loss, logits=logits, hidden_state_virus=virus_regressor_out)


# load the model
esm_config = EsmConfig.from_pretrained(MODEL_NAME, use_safetensors=True)

model = semanticESM(
    esm_config,
    MODEL_NAME,
    embed_scale_factor=embed_scale_factor,
    CSE_W=CSE_W,
    CSE_W_VIRUS_ONLY=CSE_W_VIRUS_ONLY,
    SEMANTIC_W=SEMANTIC_W,
    SEMANTIC_W_VIRUS_ONLY=SEMANTIC_W_VIRUS_ONLY,
    MAIN_W=1,
    CART_W=CART_W,
    intermediate_dim = REG_INTERMEDIATE_DIM,
    dropout = DROPOUT_REGRESSOR,
    dropout_encoder = DROPOUT_ENCODER,
    intermediate_dim_encoder = INTERMEDIATE_DIM_ENCODER,
    CSE_ALPHA = CSE_ALPHA,
    LG_W=LG_W
    )


print(model)

class CustomTrainer(Trainer):
    def __init__(
        self,
        *args,
        num_samples_per_combination=1,
        random_seed=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_samples_per_combination = num_samples_per_combination
        self.random_seed = random_seed
        self.epoch = 0  # epoch counter

    def on_epoch_begin(self):
        """ エポックごとに異なるシードを設定 """
        if self.random_seed is not None:
            epoch_seed = self.random_seed + self.epoch
            random.seed(epoch_seed)
            torch.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)
        self.epoch += 1  # Update epoch after setting the seed

    def get_train_dataloader(self):
        self.on_epoch_begin()  # Change the seed for each epoch
        train_dataset = self.train_dataset

        generator = torch.Generator()
        generator.manual_seed(self.random_seed + self.epoch)  # Different random seed for each epoch

        if isinstance(train_dataset, ConcatDataset):
            datasets = train_dataset.datasets
            subsampled_indices = []

            for dataset in datasets:
                if hasattr(dataset, "get_unique_combinations_indices"):
                    unique_combinations = dataset.get_unique_combinations_indices()
                    dataset_indices = []
                    for indices in unique_combinations.values():
                        if len(indices) <= self.num_samples_per_combination:
                            dataset_indices.extend(indices)
                        else:
                            sampled_indices = torch.randperm(len(indices), generator=generator).tolist()[:self.num_samples_per_combination]
                            dataset_indices.extend([indices[i] for i in sampled_indices])
                    subsampled_indices.extend(dataset_indices)
                else:
                    subsampled_indices.extend(range(len(dataset)))

            subsampled_dataset = Subset(train_dataset, subsampled_indices)
        else:
            unique_combinations = train_dataset.get_unique_combinations_indices()
            subsampled_indices = []

            for indices in unique_combinations.values():
                if len(indices) <= self.num_samples_per_combination:
                    subsampled_indices.extend(indices)
                else:
                    sampled_indices = torch.randperm(len(indices), generator=generator).tolist()[:self.num_samples_per_combination]
                    subsampled_indices.extend([indices[i] for i in sampled_indices])

            subsampled_dataset = Subset(train_dataset, subsampled_indices)

        batch_size = self.args.train_batch_size
        #sampler = RandomSampler(subsampled_indices, generator=generator)
        sampler = RandomSampler(subsampled_dataset, generator=generator)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=self.args.dataloader_drop_last)

        return DataLoader(
            subsampled_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

# Parameter classification
no_decay = ["bias", "LayerNorm.weight"]  # Parameters that are usually excluded from weight decay

regressor_params = []
other_params = []

for name, param in model.named_parameters():
    if "regressor" in name:
        regressor_params.append((name, param))
    else:
        other_params.append((name, param))

# Set optimizer groups
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in regressor_params],
        "weight_decay": REG_WEIGHT_DECAY,
    },
    {
        "params": [p for n, p in other_params if not any(nd in n for nd in no_decay)],
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "params": [p for n, p in other_params if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

# Define optimizer
optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)


training_args = TrainingArguments(
    output_dir=outputs_path + "results",  # output directory
    max_steps=NUM_STEPS,  # total number of training steps
    per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
    per_device_eval_batch_size=32,  # batch size for evaluation
    gradient_accumulation_steps=1,
    fp16=True,
    #learning_rate=LEARNING_RATE,
    save_strategy="steps",
    save_steps=1000,  # save every 1/10 of the total steps
    # evaluation_strategy="steps",  # evaluation strategy # may need to rename to eval_strategy if > transformers 4.46
    eval_strategy="steps",
    eval_steps=1000,  # evaluate every 1/10 of the total steps
    save_total_limit=SAVE_TOTAL_LIMIT,  # limit the number of saved checkpoints
    warmup_ratio=0.1,  # percentage of training steps to warm up LR
    #weight_decay=WEIGHT_DECAY,  # strength of weight decay
    logging_dir=outputs_path + "logs",  # directory for storing logs
    remove_unused_columns=False,
    load_best_model_at_end=True,  # load the best model at the end
)



trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train_combined,
    eval_dataset=dataset_val,
    num_samples_per_combination=1,  # Specify the number of samples per combination
    random_seed=RANDSEED,
    # save_freq=100,
)


if CHECKPOINT is not None:
    trainer.train(resume_from_checkpoint=CHECKPOINT)
else:
    trainer.train()
print("Training completed")



predictions = trainer.predict(
    dataset_test
)
print("Prediction completed")

print("Now summarizing the prediction results...")
# Convert logits to tensor if not already
logits = torch.tensor(predictions.predictions[0])

# Split logits into observed_distance and genetic_prediction
observed_distance = logits[:, 0].numpy()  # Extract the second column
cartography_distance = logits[:, 1].numpy()  # Extract the second column

selected_df_test["predicted_dist"] = (
    observed_distance  # Assign genetic_prediction to another new column
)
selected_df_test["predicted_dist_cartography"] = (
    cartography_distance  # Assign genetic_prediction to another new column
)


# summarize the prediction results
def apply_censor_cap(df, censor_col, predicted_col, score_col, output_col):
    df[output_col] = np.where(
        (df[censor_col] == 1) & (df[predicted_col] > df[score_col]),
        df[score_col],
        df[predicted_col],
    )


# Apply the function to the dataframe

apply_censor_cap(
    selected_df_test,
    "censor",
    "predicted_dist",
    "score",
    "predicted_dist_censor_cap"
)

apply_censor_cap(
    selected_df_test,
    "censor",
    "predicted_dist_cartography",
    "score",
    "predicted_dist_cartography_censor_cap",
)

latent_variables_virus = torch.tensor(predictions.predictions[1])
latent_df_virus = pd.DataFrame(latent_variables_virus)
latent_df_virus.columns = ['z1', 'z2', 'z3']

selected_df_test = selected_df_test.reset_index(drop=True)
selected_df_test[['z1', 'z2', 'z3']] = latent_df_virus[['z1', 'z2', 'z3']]


# %%
print("All processes completed")
print(f"Total time taken: {time.time() - start:.2f} seconds")


path_to_save_test_df = (
    outputs_path + f"test_df_full.csv"
)
selected_df_test.to_csv(
    path_to_save_test_df
)



import scipy.stats

corr, p_value = scipy.stats.pearsonr(selected_df_test['score'], selected_df_test['predicted_dist_censor_cap'])
print(f"Pearson correlation: {corr:.4f} (p-value: {p_value:.4g})")



#gisaid all

gisaid_df_path = storage_path + "/data/PLANT_epiflu_human_241212.csv"

# CSVを読み込んでDataFrameに格納
gisaid_df = pd.read_csv(gisaid_df_path)

gisaid_df = gisaid_df[
    ~gisaid_df["seq"].str.contains("X")
    & ~gisaid_df["seq"].str.contains("B")
    & ~gisaid_df["seq"].str.contains(r"\*")
]

gisaid_df = gisaid_df[
    gisaid_df["seq"].str.len() == 329
]




encodes_gisaid = tokenize_sequences(
    gisaid_df["seq"].tolist(), tokenizer, MAX_LENGTH
)


dataset_gisaid = TextDataset(
    encodes_gisaid
)



model = trainer.model
model.eval()
model.half()

batch_size = 64
dataloader_gisaid = DataLoader(dataset_gisaid, batch_size=batch_size, shuffle=False)

gisaid_embeddings = []

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
    for batch in dataloader_gisaid:
        input_ids_virus = batch['input_ids_virus'].to(device)
        attention_mask_virus = batch['attention_mask_virus'].to(device)

        outputs = model(input_ids_virus=input_ids_virus, attention_mask_virus=attention_mask_virus)

        gisaid_embeddings.append(outputs.hidden_state_virus.float().cpu().numpy())

gisaid_embeddings = np.concatenate(gisaid_embeddings, axis=0)

gisaid_embeddings_df = pd.DataFrame(gisaid_embeddings, columns=[f"dim{i+1}" for i in range(gisaid_embeddings.shape[1])])

gisaid_df["z1"] = gisaid_embeddings[:, 0]
gisaid_df["z2"] = gisaid_embeddings[:, 1]
gisaid_df["z3"] = gisaid_embeddings[:, 2]

print("save")
path_to_save_gisaid_df = (
    outputs_path + f"PLANT_epiflu_human_241212_with_coords.csv"
)

gisaid_df.to_csv(path_to_save_gisaid_df, index=False)
