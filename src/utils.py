import torch
import random
import numpy as np
import tarfile
import os
from src.loadData import GraphDataset
from torch_geometric.loader import DataLoader
import csv
import pandas as pd
import random
import sys
from collections import defaultdict, Counter


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def set_seed(seed=777):
    seed = seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def gzip_folder(folder_path, output_file):
    """
    Compresses an entire folder into a single .tar.gz file.
    
    Args:
        folder_path (str): Path to the folder to compress.
        output_file (str): Path to the output .tar.gz file.
    """
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    print(f"Folder '{folder_path}' has been compressed into '{output_file}'")


def compute_label_distribution(loader):

    counters = [0 for _ in range(6)]

    for batch in loader:
        for label in batch.y.tolist():
            counters[label] += 1

    return counters


def majority_vote(csv_files, output_file='majority_vote.csv'):
    pred_counts = defaultdict(list)

    # Read predictions from all CSV files
    for file in csv_files:
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            pred_counts[row['id']].append(row['pred'])

    # Compute majority vote
    final_preds = []
    for id_, preds in pred_counts.items():
        counter = Counter(preds)
        most_common = counter.most_common()
        max_count = most_common[0][1]
        top_preds = [pred for pred, count in most_common if count == max_count]
        final_pred = random.choice(top_preds)  # break ties randomly
        final_preds.append({'id': id_, 'pred': final_pred})

    # Save to output CSV
    output_df = pd.DataFrame(final_preds)
    output_df.to_csv(output_file, index=False)
    print(f"Saved majority vote results to {output_file}")