"""Attach the training-space entity table to an existing cluster cache (no retraining).

The cached `id2cluster`/`id2soft` are keyed to the *training* entity-id space,
which train.py builds over {train, valid}. The eval scripts build their ids over
{train, test}. Both enumerate train.txt first, so ids for train entities agree,
but every id >= |train| denotes a different entity in the two spaces -- indexing
the cache directly there silently attaches the wrong prototype.

`cluster_assign.load_cluster_assignments(..., entity2id=...)` now remaps by entity
string, but it needs `id2entity_nc{C}.npy` to do so. This script regenerates that
table by replaying the deterministic process_files enumeration over {train, valid}.
KMeans is not re-fit and the model is not touched.

Run from the RMPI/ directory:
    python utils/build_cluster_entity_table.py --dataset security_kg --n_clusters 64
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.cluster_assign import build_cluster_assignments  # noqa: E402


class _Params:
    """Minimal stand-in for the argparse Namespace train.py passes around."""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True,
                    help='Dataset directory under data/ (e.g. security_kg)')
    ap.add_argument('--n_clusters', type=int, default=64)
    ap.add_argument('--train_file', default='train')
    ap.add_argument('--valid_file', default='valid')
    ap.add_argument('--seed', type=int, default=41504)
    ap.add_argument('--semantic', action='store_true',
                    help='Match training: pass --semantic if training used use_semantic')
    args = ap.parse_args()

    p = _Params()
    p.main_dir = 'utils/..'  # same relative root test_ranking_P.py uses for the cache path
    p.dataset = args.dataset
    p.seed = args.seed
    p.file_paths = {
        'train': os.path.join(p.main_dir, '../data/{}/{}.txt'.format(args.dataset, args.train_file)),
        'valid': os.path.join(p.main_dir, '../data/{}/{}.txt'.format(args.dataset, args.valid_file)),
    }

    id2cluster, id2soft, n_clusters = build_cluster_assignments(
        p, n_clusters=args.n_clusters, use_semantic=args.semantic)

    print(f"[build_cluster_entity_table] {args.dataset}: "
          f"id2soft={id2soft.shape}, n_clusters={n_clusters}, "
          f"seen={(id2cluster >= 0).sum()}")


if __name__ == '__main__':
    main()
