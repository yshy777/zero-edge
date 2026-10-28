import os
import zipfile
from typing import Dict
import dgl.graphbolt as gb
import torch
import dgl
import dgl.graphbolt as gb
import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_zip_if_not_exists(zip_path: str, extract_to: str):
    # Extract zip file if the destination folder does not exist / Folder name without .zip extension
    folder_name = os.path.basename(zip_path).replace('.zip', '')
    target_dir = os.path.join(extract_to, folder_name)
    
    if not os.path.exists(target_dir):
        print(f"Extracting {zip_path} to {target_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction completed.")
    else:
        print(f"Data already extracted at {target_dir}")


def create_dataset_dict(base_dir: str):
    """Build a dictionary of dataset paths."""
    datasets = {}
    
    # Validate base path
    if not os.path.exists(base_dir):
        print(f"Path not found: {base_dir}")
        return datasets

    # Scan directory for datasets
    for dataset_name in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_name)
        
        # Check if item is a directory
        if os.path.isdir(dataset_path):
            train_f = os.path.join(dataset_path, "train.txt")
            valid_f = os.path.join(dataset_path, "valid.txt")
            test_f = os.path.join(dataset_path, "test.txt")
            
            # Verify file existence
            if os.path.exists(train_f) and os.path.exists(test_f):
                datasets[dataset_name] = { "train": train_f,
                                           "valid": valid_f,
                                           "test": test_f}
    
    # Notify if no datasets found
    if not datasets:
        print(f"Warning: No valid datasets found in {base_dir}")
        
    return datasets

        
# ## **Creating a Framework**
# ### Generate a Knowledge Graph Dataset
class KnowledgeGraphDataset(gb.OnDiskTask):
    def __init__(self, dataset_name, data_paths):

        self.dataset_name = dataset_name
        self.data_paths = data_paths

        # Process files to create datasets
        train_set, val_set, test_set = self._process_files()

        # Create DGL and CSC graphs
        self.dgl_graphs = self._create_dgl_graphs()
        self.csc_graphs = self._create_csc_graphs()
        self.set_all_nodes = self._create_set_all_nodes()

        # Create feature store
        self.feature = self._create_feature_store()

        # Set metadata
        metadata = {"name": "link_prediction", "num_classes": 2}
        super().__init__(
            validation_set=val_set,
            train_set=train_set,
            test_set=test_set,
            metadata=metadata
        )

    def _read_file(self, file_path):
        """
        Reads a file and parses it into triples (head, relation, tail).
        """
        with open(file_path, "r") as f:
            lines = f.readlines()
        triples = [tuple(line.strip().split("\t")) for line in lines]
        return triples

    def _process_files(self):
        """
        Processes the input data files and creates ItemSets for train, validation, and test sets.
        """

        dgl_graph = self._create_dgl_graphs()
        src, dst = dgl_graph['train'].edges()
        edges = torch.stack([src, dst], dim=1)
        train_set = gb.ItemSet( edges, names=("seeds",),)

        src, dst = dgl_graph['valid'].edges()
        edges = torch.stack([src, dst], dim=1)
        val_set = gb.ItemSet(
        items=(edges,
               torch.ones(len(src), dtype=torch.float64),
               torch.arange(len(src))),
               names=("seeds", "labels", "indexes"))

        src, dst = dgl_graph['test'].edges()
        edges = torch.stack([src, dst], dim=1)
        test_set = gb.ItemSet(
        items=(edges,
               torch.ones(len(src), dtype=torch.float64),
               torch.arange(len(src))),
               names=("seeds", "labels", "indexes"))

        return train_set, val_set, test_set

    def _create_dgl_graphs(self):
        """
        Creates DGL graphs for each split (train, validation, test).
        """
        entity2id, relation2id = {}, {}
        ent_id, rel_id = 0, 0
        dgl_graphs = {}

        for split, file_path in self.data_paths.items():
            triples = self._read_file(file_path)
            edges = []

            for h, r, t in triples:
                # Map entities and relations to unique IDs
                if h not in entity2id:
                    entity2id[h] = ent_id
                    ent_id += 1
                if t not in entity2id:
                    entity2id[t] = ent_id
                    ent_id += 1
                if r not in relation2id:
                    relation2id[r] = rel_id
                    rel_id += 1
                edges.append((entity2id[h], entity2id[t], relation2id[r]))

            src, dst, rel = zip(*edges)
            edges_src = torch.tensor(src, dtype=torch.int32)
            edges_dst = torch.tensor(dst, dtype=torch.int32)

            # Create DGL graph
            graph = dgl.graph((edges_src, edges_dst), num_nodes=len(entity2id))
            graph.edata["e_type"] = torch.tensor(rel, dtype=torch.int32)
            graph.edata["edge_mask"] = torch.ones(graph.num_edges(), dtype=torch.bool)
            graph.ndata["ntype"] = torch.zeros(graph.num_nodes(), dtype=torch.int32)
            graph.ndata["feat"] = torch.randn(graph.num_nodes(), 64)
            dgl_graphs[split] = graph

        return dgl_graphs

    def _create_csc_graphs(self):
        """
        Creates CSC graphs from the DGL graphs.
        """
        dgl_graph = dgl.batch([self.dgl_graphs['train'],
                               self.dgl_graphs['valid'],
                               self.dgl_graphs['test']])

        csc_graph = dgl.graphbolt.from_dglgraph(dgl_graph)
        indptr = csc_graph.csc_indptr
        indices = csc_graph.indices
        csc_graph = dgl.graphbolt.fused_csc_sampling_graph(indptr, indices)
        return csc_graph

    def _create_feature_store(self):
        dgl_graph = dgl.batch([self.dgl_graphs['train'],
                           self.dgl_graphs['valid'],
                           self.dgl_graphs['test']])

        node_features = dgl_graph.ndata["feat"]
        feature = gb.TorchBasedFeature(node_features)

        file_path = "/tmp/node_feat.npy"
        gb.numpy_save_aligned(file_path, node_features.numpy())

        feat_data = [
            gb.OnDiskFeatureData(
                domain="node",
                type=None,
                name="feat",
                format="numpy",
                path=file_path,
                in_memory=False,
            )]
        return gb.TorchBasedFeatureStore(feat_data)

    def _create_set_all_nodes(self):
      '''
      Return --> ItemSet
      '''
      train_graph = self.dgl_graphs['train']
      valid_graph = self.dgl_graphs['valid']
      test_graph = self.dgl_graphs['test']
      graph = dgl.batch([train_graph,
                        valid_graph,
                        test_graph])
      nodes = graph.nodes()
      all_nodes_set = gb.ItemSet(nodes, names="seeds")
      return all_nodes_set

    def __getitem__(self, split):
        return self.dgl_graphs[split]

    def __len__(self):
        return len(self.dgl_graphs)


# ### Graph-Based Dataset Loader for Knowledge Graphs
from functools import partial
def create_dataloader(dataset,
                      device,
                      is_train:bool,
                      batch_size:int,
                      fanouts:list,
                      negative_ratio:int):

    itemset = dataset.train_set if is_train else dataset.test_set
    datapipe = gb.ItemSampler(itemset, batch_size=256, shuffle=False)
    datapipe = datapipe.copy_to(device)

    if is_train:
        datapipe = datapipe.sample_uniform_negative(dataset.csc_graphs, negative_ratio=negative_ratio)
        datapipe = datapipe.sample_neighbor(dataset.csc_graphs, fanouts=fanouts)
        datapipe = datapipe.transform(gb.exclude_seed_edges) # Delete Positive Edge

    else:
        datapipe = datapipe.sample_uniform_negative(dataset.csc_graphs, negative_ratio=negative_ratio)
        datapipe = datapipe.sample_neighbor(dataset.csc_graphs, fanouts=fanouts) # fanouts=[-1, -1]
                                                             
    datapipe = datapipe.fetch_feature(dataset.feature, node_feature_keys=["feat"])
    return gb.DataLoader(datapipe)