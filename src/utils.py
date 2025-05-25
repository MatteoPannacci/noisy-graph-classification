import torch
import random
import numpy as np
import tarfile
import os
from src.loadData import GraphDataset
from torch_geometric.loader import DataLoader


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

    print(counters)