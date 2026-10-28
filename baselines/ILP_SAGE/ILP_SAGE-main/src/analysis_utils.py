# ## **Inductive And Transductive Data Set Analysis**
# ### **Node Degree % by Split**

#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import sklearn
import sys
from IPython.display import clear_output
from tqdm import tqdm
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
from src.logger import save_df_to_excel
from collections import defaultdict


def get_degrees(base_dir):
    datasets = sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))],
        key=lambda x: (x.lower().endswith('_ind'), x.lower()))
    
    data = []
    for d in datasets:
        row = {'Dataset': d}
        for split in ['train.txt', 'test.txt']:
            path = os.path.join(base_dir, d, split)
            if os.path.exists(path):
                df = pd.read_csv(path, sep="\t", header=None, names=["head", "relation", "tail"])
                G = nx.DiGraph()
                G.add_edges_from(zip(df["head"], df["tail"]))
                deg = dict(G.degree())
                total = len(deg) or 1
                for k in [1, 2, 3]:
                    row[f'Deg{k}_%_{split}'] = round(100 * sum(v == k for v in deg.values()) / total, 2)
        data.append(row)
    return pd.DataFrame(data)

# ## **Graph Structure Summary: Averaged Node Degrees**
def compute_degree_stats(file_path, directed=True):
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path, sep="\t", header=None, names=["head", "relation", "tail"])
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_edges_from(zip(df["head"], df["tail"]))
        degrees = dict(G.degree())
        total_degree = sum(degrees.values())
        num_nodes = len(degrees)
        deg_counts = {i: sum(1 for d in degrees.values() if d == i) for i in range(1, 5)}
        deg_percents = {
            f"deg{i}_percent": round((deg_counts[i] / num_nodes) * 100, 2) if num_nodes > 0 else 0
            for i in range(1, 5)
        }
        return {
            "avg_degree": round(total_degree / num_nodes, 2) if num_nodes > 0 else 0,
            "total_degree": total_degree,
            "num_nodes": num_nodes,
            **deg_percents
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_and_group_datasets(base_dir):
    datasets = sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))],
        key=lambda x: (x.lower().endswith('_ind'), x.lower())
    )
    grouped = defaultdict(lambda: defaultdict(list))
    for dataset in datasets:
        base_name = dataset.split('_')[0]
        dataset_path = os.path.join(base_dir, dataset)
        train_path = os.path.join(dataset_path, 'train.txt')
        test_path = os.path.join(dataset_path, 'test.txt')
        train_stats = compute_degree_stats(train_path)
        test_stats = compute_degree_stats(test_path)
        if train_stats and test_stats:
            total_combined = train_stats["total_degree"] + test_stats["total_degree"]
            nodes_combined = train_stats["num_nodes"] + test_stats["num_nodes"]
            combined_avg = round(total_combined / nodes_combined, 2) if nodes_combined > 0 else 0
            for i in range(1, 5):
                grouped[base_name][f"Train Deg{i}%"].append(train_stats[f"deg{i}_percent"])
                grouped[base_name][f"Test Deg{i}%"].append(test_stats[f"deg{i}_percent"])
            grouped[base_name]["Train AvgDeg"].append(train_stats["avg_degree"])
            grouped[base_name]["Test AvgDeg"].append(test_stats["avg_degree"])
            grouped[base_name]["Overall AvgDeg"].append(combined_avg)

    final_rows = []
    for group, stats in grouped.items():
        row = {"Dataset": group}
        for key, values in stats.items():
            row[key] = round(sum(values) / len(values), 2)
        final_rows.append(row)

    df = pd.DataFrame(final_rows)
    avg_cols = ["Train AvgDeg", "Test AvgDeg", "Overall AvgDeg"]
    percent_cols = [f"{split} Deg{i}%" for split in ["Train", "Test"] for i in range(1, 5)]
    ordered_cols = ["Dataset"] + avg_cols + percent_cols
    df = df[[col for col in ordered_cols if col in df.columns]]
    df = df.sort_values(by='Dataset', key=lambda s: s.str.lower().str.endswith('_ind').astype(int).astype(str) + s.str.lower()).reset_index(drop=True)
    return df


# ## **Node Degree Summary**
def get_all_degrees(base_dir_1, base_dir_2):
    def process_directory(base_dir):
        datasets = sorted(
            [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))],
            key=lambda x: (x.lower().endswith('_ind'), x.lower())
        )
        results = []
        for dataset in datasets:
            row = {'Dataset': dataset}
            for split in ['train.txt', 'test.txt']:
                path = os.path.join(base_dir, dataset, split)
                if os.path.exists(path):
                    df = pd.read_csv(path, sep="\t", header=None, names=["head", "relation", "tail"])
                    G = nx.DiGraph()
                    G.add_edges_from(zip(df["head"], df["tail"]))
                    deg = dict(G.degree())
                    for k in [1, 2, 3]:
                        row[f'Deg_{k}_{split}'] = sum(v == k for v in deg.values())
            results.append(row)
        return pd.DataFrame(results)

    df1 = process_directory(base_dir_1)
    df2 = process_directory(base_dir_2)
    combined_df = df1.set_index('Dataset').add(df2.set_index('Dataset'), fill_value=0).reset_index()

    # Final sort: non-_ind datasets first
    combined_df = combined_df.sort_values(
        by='Dataset',
        key=lambda s: s.str.lower().str.endswith('_ind').astype(int).astype(str) + s.str.lower()).reset_index(drop=True)
    return combined_df