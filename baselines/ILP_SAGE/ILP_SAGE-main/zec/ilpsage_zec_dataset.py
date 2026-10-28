# zec/ilpsage_zec_dataset.py — ILP-SAGE KnowledgeGraphDataset 的 ZEC 适配版
#
# 与原版 src/data_utils.py 的两处差异（其余逐行保持官方实现）：
#   1. csc 采样图只用 train 边（原版 dgl.batch(train+valid+test) 会把 test 边漏进
#      训练负采样与测试评分子图 —— ZEC 纪律：冷头在训练图中必须零边，且评分子图
#      不得含 test 边）。
#   2. 特征缓存路径 /tmp/node_feat.npy → tempfile（Windows 无 /tmp）。
#   3. 缺少 valid.txt 时跳过（官方 create_dataset_dict 只查 train/test 存在性，
#      但 _create_dgl_graphs 会无条件读 valid —— 原版只能跑三文件齐全的数据集）。
import os
import tempfile

import dgl
import dgl.graphbolt as gb
import numpy as np
import torch


class KnowledgeGraphDatasetZEC(gb.OnDiskTask):
    def __init__(self, dataset_name, data_paths, node_feat=None):
        self.dataset_name = dataset_name
        self.data_paths = {k: v for k, v in data_paths.items() if v and os.path.exists(v)}
        # ZEC 补丁：node_feat 允许外部注入 train 图节点特征（rank 时复用 train 时
        # 存进 ckpt 的同一份随机特征 —— 原实现 train/rank 各 randn 一份，特征不
        # 匹配会让 rank 的 CWE 嵌入相对 predictor 全成垃圾）。
        self.node_feat = node_feat
        self.dgl_graphs = self._create_dgl_graphs()
        train_set, val_set, test_set = self._process_files()
        self.csc_graph = self._create_csc_graph()
        self.set_all_nodes = self._create_set_all_nodes()
        self.feature = self._create_feature_store()
        metadata = {"name": "link_prediction", "num_classes": 2}
        super().__init__(
            validation_set=val_set,
            train_set=train_set,
            test_set=test_set,
            metadata=metadata)

    def _read_file(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        triples = [tuple(line.strip().split("\t")) for line in lines]
        return triples

    def _process_files(self):
        dgl_graph = self.dgl_graphs
        edges = []
        for split in ("train", "valid", "test"):
            if split in dgl_graph:
                src, dst = dgl_graph[split].edges()
                edges.append((split, torch.stack([src, dst], dim=1)))
        train_set = gb.ItemSet(
            next(e for s, e in edges if s == "train"), names=("seeds",))
        val_set = test_set = None
        if any(s == "valid" for s, _ in edges):
            e = next(e for s, e in edges if s == "valid")
            val_set = gb.ItemSet(
                items=(e, torch.ones(len(e), dtype=torch.float64),
                       torch.arange(len(e))),
                names=("seeds", "labels", "indexes"))
        if any(s == "test" for s, _ in edges):
            e = next(e for s, e in edges if s == "test")
            test_set = gb.ItemSet(
                items=(e, torch.ones(len(e), dtype=torch.float64),
                       torch.arange(len(e))),
                names=("seeds", "labels", "indexes"))
        return train_set, val_set, test_set

    def _create_dgl_graphs(self):
        # ZEC 补丁：先全 split 建全局 ID，再建各 split 图。所有图 num_nodes=全局实体数
        # （原版边建 ID 边建图，train 图只覆盖 train 实体，test-only 冷实体 ID 越界 ——
        #  原版靠 csc batch 三图掩盖了这一点；ZEC 的冷 head 全是 test-only，必须全局）。
        # 冷实体在 train csc 中为零度节点：采样空邻域 → 嵌入仅剩特征变换、无图信号，
        # 与 ZEC「训练图中零边」的语义一致（边仍只来自 train，无泄漏）。
        entity2id, relation2id = {}, {}
        ent_id, rel_id = 0, 0
        split_edges = {}
        for split, file_path in self.data_paths.items():
            triples = self._read_file(file_path)
            edges = []
            for h, r, t in triples:
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
            split_edges[split] = edges
        self.entity2id = entity2id
        self.relation2id = relation2id
        num_nodes = len(entity2id)
        dgl_graphs = {}
        for split, edges in split_edges.items():
            src, dst, rel = zip(*edges)
            edges_src = torch.tensor(src, dtype=torch.int32)
            edges_dst = torch.tensor(dst, dtype=torch.int32)
            graph = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)
            graph.edata["e_type"] = torch.tensor(rel, dtype=torch.int32)
            graph.edata["edge_mask"] = torch.ones(graph.num_edges(), dtype=torch.bool)
            graph.ndata["ntype"] = torch.zeros(num_nodes, dtype=torch.int32)
            graph.ndata["feat"] = (
                torch.from_numpy(self.node_feat).float()
                if (split == "train" and self.node_feat is not None)
                else torch.randn(num_nodes, 64))
            dgl_graphs[split] = graph
        return dgl_graphs

    def _create_csc_graph(self):
        # ZEC 补丁：只取 train 图（原版 batch 三图）
        csc_graph = dgl.graphbolt.from_dglgraph(self.dgl_graphs["train"])
        indptr = csc_graph.csc_indptr
        indices = csc_graph.indices
        csc_graph = dgl.graphbolt.fused_csc_sampling_graph(indptr, indices)
        return csc_graph

    def _create_feature_store(self):
        dgl_graph = self.dgl_graphs["train"]
        node_features = dgl_graph.ndata["feat"]
        feature = gb.TorchBasedFeature(node_features)
        # ZEC 补丁：/tmp 在 Windows 不存在；dgl 2.2.1 已删除 gb.numpy_save_aligned，
        # 2.2.1 的 OnDiskFeatureData(format="numpy") 用标准 np.load，普通 .npy 即可。
        file_path = os.path.join(tempfile.gettempdir(), "ilpsage_node_feat.npy")
        np.save(file_path, node_features.numpy())
        feat_data = [
            gb.OnDiskFeatureData(
                domain="node", type=None, name="feat", format="numpy",
                path=file_path, in_memory=False)]
        return gb.TorchBasedFeatureStore(feat_data)

    def _create_set_all_nodes(self):
        graph = self.dgl_graphs["train"]
        nodes = graph.nodes()
        all_nodes_set = gb.ItemSet(nodes, names="seeds")
        return all_nodes_set

    def __getitem__(self, split):
        return self.dgl_graphs[split]

    def __len__(self):
        return len(self.dgl_graphs)


def create_train_dataloader(dataset, device, batch_size, fanouts, negative_ratio):
    """官方 create_dataloader 的训练分支（is_train=True），逐行保持官方实现。"""
    itemset = dataset.train_set
    datapipe = gb.ItemSampler(itemset, batch_size=256, shuffle=False)
    datapipe = datapipe.copy_to(device)
    datapipe = datapipe.sample_uniform_negative(dataset.csc_graph, negative_ratio=negative_ratio)
    datapipe = datapipe.sample_neighbor(dataset.csc_graph, fanouts=fanouts)
    datapipe = datapipe.transform(gb.exclude_seed_edges)
    datapipe = datapipe.fetch_feature(dataset.feature, node_feature_keys=["feat"])
    return gb.DataLoader(datapipe)


def create_score_dataloader(dataset, device, seed_pairs, batch_size, fanouts):
    """ZEC 评分分支：种子对由外部给定（每 test link 51 个候选），无内部负采样。
    其余与官方 test 分支一致（sample_neighbor + fetch_feature）。
    dtype 必须与 csc 图 indices 一致（官方训练分支靠 sample_uniform_negative
    中途转 dtype；本分支没有它，需显式对齐，否则 neighbor_sampler 断言失败）。"""
    seed_dtype = dataset.csc_graph.indices.dtype
    itemset = gb.ItemSet(
        torch.tensor(seed_pairs, dtype=seed_dtype), names=("seeds",))
    datapipe = gb.ItemSampler(itemset, batch_size=batch_size, shuffle=False)
    datapipe = datapipe.copy_to(device)
    datapipe = datapipe.sample_neighbor(dataset.csc_graph, fanouts=fanouts)
    datapipe = datapipe.fetch_feature(dataset.feature, node_feature_keys=["feat"])
    return gb.DataLoader(datapipe)
