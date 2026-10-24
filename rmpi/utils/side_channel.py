"""通用侧信道加载器（C1）：替代 datasets.load_product_fingerprint。

把「图外属性 -> 每实体固定维向量」的读取从 security-KG 特化（CPE 产品指纹 JSON）
解耦成通用接口，供标准 KG（实体名 char n-gram，见 side_channel_text.py）复用。

三种输入格式，按优先级：
  1. side_channel.npz + side_channel_entities.txt —— 行序 = 实体名列表（推荐，省内存）
  2. product_fingerprint.json —— 旧格式 {entity_name: [vec]}（向后兼容 security-KG）
  3. 均缺 -> (None, 0)

签名与 load_product_fingerprint(fp_path, id2entity) 完全一致，可直接 drop-in。
"""
import json
import os

import numpy as np
import scipy.sparse as sp


def _dense(mat, n_ent, D):
    arr = np.zeros((n_ent, D), dtype=np.float32)
    if sp.issparse(mat):
        mat = mat.toarray()
    if mat.shape[0] == n_ent and mat.shape[1] == D:
        arr[:] = mat
    return arr


def load_side_channel(sc_path, id2entity):
    """sc_path: side_channel.npz 路径。返回 (entity_matrix [n_entities, D], D)。

    - npz + 同名 _entities.txt：按实体名映射回全局 id，缺名 -> 零向量。
    - 退化为 product_fingerprint.json（旧 security-KG 格式）。
    - 都没有 -> (None, 0)。
    """
    n_ent = max(id2entity) + 1

    # 格式 1：npz + entities.txt
    if os.path.exists(sc_path):
        mat = sp.load_npz(sc_path)
        D = mat.shape[1]
        entities_path = os.path.splitext(sc_path)[0] + "_entities.txt"
        row_of = {}
        if os.path.exists(entities_path):
            with open(entities_path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    name = line.rstrip("\n")
                    if name:
                        row_of[name] = i
        arr = np.zeros((n_ent, D), dtype=np.float32)
        mat = mat.toarray()
        for eid, name in id2entity.items():
            r = row_of.get(name)
            if r is not None and r < mat.shape[0]:
                arr[eid] = mat[r]
        return arr, D

    # 格式 2：旧 product_fingerprint.json（security-KG CPE）
    legacy = os.path.join(os.path.dirname(sc_path), "product_fingerprint.json")
    if os.path.exists(legacy):
        with open(legacy, encoding="utf-8") as f:
            fp = json.load(f)
        if fp:
            D = len(next(iter(fp.values())))
            arr = np.zeros((n_ent, D), dtype=np.float32)
            for eid, name in id2entity.items():
                if name in fp:
                    arr[eid] = np.asarray(fp[name], dtype=np.float32)
            return arr, D

    return None, 0
