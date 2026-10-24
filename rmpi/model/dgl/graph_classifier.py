import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np



class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id
        self.relation_list = list(self.relation2id.values())
        self.link_mode = 6
        self.is_big_dataset = getattr(self.params, 'is_big_dataset', False)


        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)

        torch.nn.init.normal_(self.rel_emb.weight)

        self.fc_reld1 = nn.ModuleList([nn.Linear(self.params.rel_emb_dim, self.params.rel_emb_dim, bias=True)
                                      for _ in range(6)
                                      ])
        self.fc_reld2 = nn.ModuleList([nn.Linear(self.params.rel_emb_dim, self.params.rel_emb_dim, bias=True)
                                      for _ in range(6)
                                      ])
        self.fc_reld = nn.Linear(self.params.rel_emb_dim, self.params.rel_emb_dim, bias=True)

        self.fc_layer = nn.Linear(self.params.rel_emb_dim, 1)

        if self.params.conc:
            self.conc = nn.Linear(self.params.rel_emb_dim*2, self.params.rel_emb_dim)

        if self.params.gpu >= 0:
            self.device = torch.device('cuda:%d' % self.params.gpu)
        else:
            self.device = torch.device('cpu')


        self.leakyrelu = nn.LeakyReLU(0.2)
        self.drop = torch.nn.Dropout(0.5)

        # ProtoRMPI: cluster-prototype injection (ablation >= 2).
        self.n_clusters = getattr(self.params, 'n_clusters', 0)
        self.use_proto = getattr(self.params, 'ablation', 0) >= 2
        # Per-CWE tail embedding: the CWE answer entities are SEEN at training
        # time (only the CVE query is unseen), so give them a learnable lookup
        # instead of collapsing them onto a shared cluster prototype. Index 0 =
        # non-CWE/OOV. cwe2id is saved with the model so eval remaps by name.
        self.cwe2id = getattr(self.params, 'cwe2id', {})
        self.n_cwes = len(self.cwe2id)
        if self.use_proto:
            # +1 row for the "unknown/unseen" cluster (hard label -1 falls here).
            self.prototype_emb = nn.Embedding(self.n_clusters + 1, self.params.rel_emb_dim, sparse=False)
            torch.nn.init.normal_(self.prototype_emb.weight)
            if self.n_cwes > 0:
                self.cwe_emb = nn.Embedding(self.n_cwes + 1, self.params.rel_emb_dim, sparse=False)
                torch.nn.init.normal_(self.cwe_emb.weight)
                # concat([g_rep, head_slot, tail_proto, tail_cwe_emb]) -> project back
                # (head_slot fed zeros since the C6 head cut; width kept for checkpoint compat)
                self.fc_proto = nn.Linear(self.params.rel_emb_dim * 4, self.params.rel_emb_dim, bias=True)
            else:
                # concat([g_rep, head_slot, tail_proto]) -> project back to rel_emb_dim
                self.fc_proto = nn.Linear(self.params.rel_emb_dim * 3, self.params.rel_emb_dim, bias=True)

        # Scheme 3: product-family fingerprint injection (CVE node identity).
        # concat([g_rep, head_fp, tail_fp]) -> project back to rel_emb_dim.
        self.prod_dim = getattr(self.params, 'prod_dim', 0)
        if self.prod_dim > 0:
            self.fc_prod = nn.Linear(self.params.rel_emb_dim + 2 * self.prod_dim,
                                     self.params.rel_emb_dim, bias=True)


    def rel_aggr(self, graph, u_node, v_node, num_nodes, num_edges, aggr_flag, is_drop):
        # print("node pair:", u_node, v_node)
        u_in_edge = graph.in_edges(u_node, 'all')
        u_out_edge = graph.out_edges(u_node, 'all')
        v_in_edge = graph.in_edges(v_node, 'all')
        v_out_edge = graph.out_edges(v_node, 'all')

        edge_mask = self.drop(torch.ones(num_edges, device=self.device))



        in_edge_out = torch.sparse_coo_tensor(torch.cat((u_in_edge[1].unsqueeze(0), u_in_edge[2].unsqueeze(0)), 0),
                                              torch.ones(len(u_in_edge[2]), device=self.device), size=torch.Size((num_nodes, num_edges)))
        out_edge_out = torch.sparse_coo_tensor(torch.cat((u_out_edge[0].unsqueeze(0), u_out_edge[2].unsqueeze(0)), 0),
                                               torch.ones(len(u_out_edge[2]), device=self.device), size=torch.Size((num_nodes, num_edges)))
        in_edge_in = torch.sparse_coo_tensor(torch.cat((v_in_edge[1].unsqueeze(0), v_in_edge[2].unsqueeze(0)), 0),
                                             torch.ones(len(v_in_edge[2]), device=self.device), size=torch.Size((num_nodes, num_edges)))
        out_edge_in = torch.sparse_coo_tensor(torch.cat((v_out_edge[0].unsqueeze(0), v_out_edge[2].unsqueeze(0)), 0),
                                              torch.ones(len(v_out_edge[2]), device=self.device), size=torch.Size((num_nodes, num_edges)))

        if is_drop:
            in_edge_out = self.sparse_dropout(in_edge_out, edge_mask)
            out_edge_out = self.sparse_dropout(out_edge_out, edge_mask)
            in_edge_in = self.sparse_dropout(in_edge_in, edge_mask)
            out_edge_in = self.sparse_dropout(out_edge_in, edge_mask)

        if self.is_big_dataset:  # smaller memory
            in_edge_out = self.sparse_index_select(in_edge_out, u_node).to(device=self.device)
            out_edge_out = self.sparse_index_select(out_edge_out, u_node).to(device=self.device)
            in_edge_in = self.sparse_index_select(in_edge_in, v_node).to(device=self.device)
            out_edge_in = self.sparse_index_select(out_edge_in, v_node).to(device=self.device)
        else:  # faster calculation
            in_edge_out = in_edge_out.to(device=self.device).to_dense()[u_node].to_sparse()
            out_edge_out = out_edge_out.to(device=self.device).to_dense()[u_node].to_sparse()
            in_edge_in = in_edge_in.to(device=self.device).to_dense()[v_node].to_sparse()
            out_edge_in = out_edge_in.to(device=self.device).to_dense()[v_node].to_sparse()


        edge_mode_5 = out_edge_out.mul(in_edge_in)
        edge_mode_6 = in_edge_out.mul(out_edge_in)
        out_edge_out = out_edge_out.sub(edge_mode_5)
        in_edge_in = in_edge_in.sub(edge_mode_5)
        in_edge_out = in_edge_out.sub(edge_mode_6)
        out_edge_in = out_edge_in.sub(edge_mode_6)


        if aggr_flag == 1:
            edge_connect_l = [in_edge_out, out_edge_out, in_edge_in, out_edge_in, edge_mode_5, edge_mode_6]

            rel_neighbor_embd = sum([edge_connect_l[i] @ self.fc_reld2[i](self.h1) for i in range(self.link_mode)])


            return rel_neighbor_embd

        elif aggr_flag == 2:
            edge_connect_l = [in_edge_out, out_edge_out, in_edge_in, out_edge_in, edge_mode_5, edge_mode_6]

            if self.params.target2nei_atten:
                xxx = self.rel_emb(self.neighbor_edges2rels)
                rel_2directed_atten = torch.einsum('bd,nd->bn', [xxx, self.h0])
                rel_2directed_atten = self.leakyrelu(rel_2directed_atten)

                item = list()
                for i in range(6):
                    atten = self.sparse_dense_mul(edge_connect_l[i], rel_2directed_atten).to_dense()
                    mask = (atten == 0).bool()
                    atten_softmax = torch.nn.Softmax(dim=-1)(atten.masked_fill(mask, -np.inf))
                    atten_softmax = torch.where(torch.isnan(atten_softmax), torch.full_like(atten_softmax, 0),
                                                atten_softmax).to_sparse()
                    agg_i = atten_softmax @ self.fc_reld1[i](self.h0)
                    item.append(agg_i)
                rel_neighbor_embd = sum(item)

            else:
                rel_neighbor_embd = sum([edge_connect_l[i] @ self.fc_reld1[i](self.h0) for i in range(self.link_mode)])

            return rel_neighbor_embd

        elif aggr_flag == 0:
            # # print(edge_mode_5)
            num_target = u_node.shape[0]
            dis_target_edge_ids = self.rel_edge_ids
            self_mask = torch.ones((num_target, num_edges))
            for i in range(num_target):
                self_mask[i][dis_target_edge_ids[i]] = 0
            self_mask = self_mask.to(device=self.device)
            edge_mode_5 = self.sparse_dense_mul(edge_mode_5, self_mask)


            edge_connect_l = in_edge_out + out_edge_out + in_edge_in + out_edge_in + edge_mode_5 + edge_mode_6

            neighbor_rel_embeds = self.rel_emb(graph.edata['type'])

            rel_2directed_atten = torch.einsum('bd,nd->bn', [self.fc_reld(self.rel_emb(self.rel_labels)), self.fc_reld(neighbor_rel_embeds)])
            rel_2directed_atten = self.leakyrelu(rel_2directed_atten)

            atten = self.sparse_dense_mul(edge_connect_l, rel_2directed_atten).to_dense()
            mask = (atten == 0).bool()
            atten_softmax = torch.nn.Softmax(dim=-1)(atten.masked_fill(mask, -np.inf))
            atten_softmax = torch.where(torch.isnan(atten_softmax), torch.full_like(atten_softmax, 0),
                                            atten_softmax).to_sparse()
            rel_neighbor_embd = atten_softmax @ self.fc_reld(neighbor_rel_embeds)


            return rel_neighbor_embd


    def forward(self, data):

        en_g, dis_g, rel_labels = data

        # relational aggregation begin
        self.rel_labels = rel_labels
        num_nodes = en_g.num_nodes()
        num_edges = en_g.num_edges()

        head_ids = (en_g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (en_g.ndata['id'] == 2).nonzero().squeeze(1)

        head_node, tail_node = head_ids, tail_ids
        u_in_nei = en_g.in_edges(head_node, 'all')
        u_out_nei = en_g.out_edges(head_node, 'all')
        v_in_nei = en_g.in_edges(tail_node, 'all')
        v_out_nei = en_g.out_edges(tail_node, 'all')

        self.h0 = self.rel_emb(en_g.edata['type'])

        neighbor_edges = torch.cat((u_in_nei[2], u_out_nei[2], v_in_nei[2], v_out_nei[2]))
        neighbor_edges = torch.unique(neighbor_edges, sorted=False)

        # edge2rel (neighbor edge -> target relation label) is only consumed by the
        # target-aware 2-hop attention path in rel_aggr(aggr_flag=2). When that is
        # off (the default), this per-pair .tolist() dict loop is dead O(nnz) Python
        # work on the giant full-scale subgraphs, so it is skipped entirely
        # (zero-result-change).
        if self.params.target2nei_atten:
            edge2rel = dict()
            for i in range(len(rel_labels)):
                u_node_i = head_node[i]
                v_node_i = tail_node[i]
                u_i_in_edge = en_g.in_edges(u_node_i, 'all')[2]
                u_i_out_edge = en_g.out_edges(u_node_i, 'all')[2]
                v_i_in_edge = en_g.in_edges(v_node_i, 'all')[2]
                v_i_out_edge = en_g.out_edges(v_node_i, 'all')[2]
                i_neighbor_edges = torch.cat((u_i_in_edge, u_i_out_edge, v_i_in_edge, v_i_out_edge))
                i_neighbor_edges = torch.unique(i_neighbor_edges, sorted=False)
                for eid in i_neighbor_edges.cpu().numpy().tolist():
                    edge2rel[eid] = rel_labels[i]
            neighbor_edges2rels = [edge2rel[eid] for eid in neighbor_edges.cpu().numpy().tolist()]
            neighbor_edges2rels = torch.Tensor(neighbor_edges2rels).long().to(device=self.device)
            self.neighbor_edges2rels = neighbor_edges2rels

        neighbor_u_nodes = en_g.edges()[0][neighbor_edges]
        neighbor_v_nodes = en_g.edges()[1][neighbor_edges]

        self.neighbor_edges = neighbor_edges


        '''
        agg_flag:
            2: 2-hop neighbors
            1: 1-hop directed neighbors
            0: 1-hop disclosing directed neighbors
        '''
        self.h0_extracted = self.h0[neighbor_edges]
        h_0_N = self.rel_aggr(en_g, neighbor_u_nodes, neighbor_v_nodes, num_nodes, num_edges, aggr_flag=2, is_drop=True)
        h_0_N = F.relu(h_0_N)
        self.h1 = self.rel_emb(en_g.edata['type'])

        # Scatter-add the 2-hop neighbor embeddings into h1, vectorized (the
        # previous per-edge Python loop was O(nnz)). neighbor_edges is unique, so
        # index_add_ writes each row exactly once.
        self.h1.index_add_(0, neighbor_edges, h_0_N)


        # One target edge per subgraph = the (head, tail) edge (DGL 0.4's
        # `edge_id` picked one per pair; DGL 1.x `edge_ids` returns the first
        # parallel edge for each pair). Marking all (0,1) edges via edata['id']
        # breaks on multi-relation (h,t) pairs (17 vs 16 shape mismatch).
        rel_edge_ids = en_g.edge_ids(head_ids, tail_ids).tolist()

        self.h1_extracted = self.h1[rel_edge_ids]
        self.rel_edge_ids = rel_edge_ids
        self.rel_edge_ids = rel_edge_ids
        h_1_N = self.rel_aggr(en_g, head_node, tail_node, num_nodes, num_edges, aggr_flag=1, is_drop=True)

        h_1_N = F.relu(h_1_N)
        h2 = self.h1_extracted+h_1_N



        if self.params.ablation == 0: # RMP base
            final_embed = h2
            g_rep = F.normalize(final_embed, p=2, dim=-1)
        elif self.params.ablation >= 1:  # RMP-NE and prototype variants (ablation >= 1) use entity aggregation
            # # entity aggregation begin
            dis_head_ids = (dis_g.ndata['id'] == 1).nonzero().squeeze(1)
            dis_tail_ids = (dis_g.ndata['id'] == 2).nonzero().squeeze(1)

            dis_num_nodes = dis_g.num_nodes()
            dis_num_edges = dis_g.num_edges()
            one_hop_nei_embd = self.rel_aggr(dis_g, dis_head_ids, dis_tail_ids, dis_num_nodes, dis_num_edges,
                                             aggr_flag=0, is_drop=True)
            one_hop_nei_embd = F.relu(one_hop_nei_embd)
            # #
            if self.params.conc:
                h2 = F.normalize(h2, p=2, dim=-1)
                one_hop_nei_embd = F.normalize(one_hop_nei_embd, p=2, dim=-1)
                g_rep = self.conc(torch.cat([h2, one_hop_nei_embd], dim=1))
            else:
                final_embed = h2 + one_hop_nei_embd
                g_rep = F.normalize(final_embed, p=2, dim=-1)
        # entity aggregation begin


        if getattr(self, 'use_proto', False):
            # 位置 B: concat head/tail cluster prototypes to g_rep before scoring.
            tail_cluster = en_g.ndata['cluster'][tail_ids]
            if self.params.ablation == 2:
                # + hard cluster prototype (learnable per-cluster embedding)
                # Unseen entities (cluster == -1, e.g. long-tail nodes that only
                # appear in valid/test edges but are pulled into train subgraphs
                # as 2-hop neighbors) would index out of range; fall back to
                # cluster 0 so training does not crash.
                tail_proto = self.prototype_emb(tail_cluster.clamp(min=0))
            else:
                # + soft cluster prototype (= ProtoRMPI): soft assignment alpha
                #   over the K learnable prototypes. alpha comes from the fixed
                #   KMeans geometry (cluster_assign.py); the prototype embeddings
                #   are learned end-to-end in the relation-embedding space.
                tail_soft = en_g.ndata['soft'][tail_ids]
                tail_proto = tail_soft @ self.prototype_emb.weight[:self.n_clusters]
            # Head-side prototype cut (C6): in the ZEC regime every cold head
            # maps to the same prototype slot (hard: cluster -1 -> 0; soft: one
            # shared zero-feature assignment), so the head-side prototype is a
            # constant with zero per-query discriminative power. Its concat slot
            # is fed zeros while fc_proto keeps its trained width (old
            # checkpoints load unchanged); the cut drops only the trained
            # constant offset (audit: head-prototype discrimination = 0).
            head_side = torch.zeros_like(tail_proto)
            if getattr(self, 'n_cwes', 0) > 0:
                if 'cwe' in en_g.ndata:
                    tail_cwe = en_g.ndata['cwe'][tail_ids].long().clamp(min=0)
                else:
                    tail_cwe = torch.zeros_like(tail_ids, dtype=torch.long)
                tail_cwe_emb = self.cwe_emb(tail_cwe)
                g_rep = self.fc_proto(torch.cat([g_rep, head_side, tail_proto, tail_cwe_emb], dim=-1))
            else:
                g_rep = self.fc_proto(torch.cat([g_rep, head_side, tail_proto], dim=-1))
            g_rep = F.normalize(g_rep, p=2, dim=-1)

        # Scheme 3: inject head/tail product-family fingerprints (CVE identity).
        # 'prod' in ndata guards old eval scripts that never attach the feature.
        # getattr guards old checkpoints saved before __init__ gained prod_dim.
        if getattr(self, 'prod_dim', 0) > 0 and 'prod' in en_g.ndata:
            head_fp = en_g.ndata['prod'][head_ids]
            tail_fp = en_g.ndata['prod'][tail_ids]
            g_rep = self.fc_prod(torch.cat([g_rep, head_fp, tail_fp], dim=-1))
            g_rep = F.normalize(g_rep, p=2, dim=-1)

        output = self.fc_layer(g_rep)
        return output

    @staticmethod
    def sparse_dense_mul(s, d):
        s = s.coalesce()
        i = s.indices()
        v = s.values()
        dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
        return torch.sparse_coo_tensor(i, v * dv, s.size())

    @staticmethod
    def sparse_dropout(s, mask):
        """Apply a 1-D per-edge dropout mask `mask` (shape [num_edges]) to the
        values of sparse tensor `s` (shape [num_nodes, num_edges]), scaling each
        value by the mask of its column (edge) index. Mathematically identical
        to `sparse_dense_mul(s, mask.repeat(num_nodes, 1))` but avoids
        materialising the (num_nodes x num_edges) dense mask, which OOMs on the
        large ranking subgraphs of the security KG."""
        s = s.coalesce()
        i = s.indices()
        v = s.values() * mask[i[1, :]]
        return torch.sparse_coo_tensor(i, v, s.size())

    @staticmethod
    def sparse_index_select(s, idx):
        # Select rows of sparse tensor `s` by node ids `idx` (remapping them to
        # 0..len(idx)-1), vectorized. Mathematically identical to the previous
        # per-node Python loop, which was O(len(idx) x nnz) plus O(n^2) torch.cat
        # accumulation and dominated the giant full-scale subgraphs (5k-25x too
        # slow). This version is O(nnz) and preserves duplicate ids in `idx`
        # (aggr_flag=2 passes neighbor_u_nodes with repeated hub nodes) via
        # searchsorted + repeat_interleave.
        s = s.coalesce()
        rows = s.indices()[0]
        cols = s.indices()[1]
        idx = idx.long()
        idx_sorted, sort_order = torch.sort(idx)
        left = torch.searchsorted(idx_sorted, rows)
        right = torch.searchsorted(idx_sorted, rows, right=True)
        cnt = right - left                       # occurrences of each row in idx
        keep = cnt > 0
        cnt, cols, left = cnt[keep], cols[keep], left[keep]
        new_cols = torch.repeat_interleave(cols, cnt)
        base = torch.repeat_interleave(left, cnt)
        group_start = torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        offsets = torch.arange(cnt.sum(), device=rows.device) - group_start
        new_rows = sort_order[base + offsets]
        indices_new = torch.stack([new_rows, new_cols], 0)
        return torch.sparse_coo_tensor(indices_new,
                                       torch.ones(new_rows.numel(), device=rows.device),
                                       torch.Size((len(idx), s.shape[1])))
