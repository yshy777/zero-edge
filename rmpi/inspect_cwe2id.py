"""Count the cwe2id table saved in the 2+SC production checkpoint (solves the '796 classes' audit question)."""
import argparse
import os
import torch

_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'expri_save_models', 'security_kg_full_ind_RMPI_proto_hard_sc',
                        'best_graph_classifier.pth')
p = argparse.ArgumentParser()
p.add_argument('--ckpt', default=_DEFAULT)
ckpt = p.parse_args().ckpt
m = torch.load(ckpt, weights_only=False)
cwe2id = getattr(m, 'cwe2id', {})
print('n_cwes attr:', getattr(m, 'n_cwes', None))
print('len(cwe2id):', len(cwe2id))
names = sorted(cwe2id.keys())
print('first:', names[:3])
print('last:', names[-3:])
# check id assignment
ids = sorted(cwe2id.values())
print('min id:', ids[0], 'max id:', ids[-1], 'contiguous:', ids == list(range(1, len(ids) + 1)))
