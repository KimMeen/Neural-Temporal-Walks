import time
import numpy as np
import torch
import multiprocessing as mp
import torch.nn as nn
from utils import *
from position import *
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, non_linear=True):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

        self.non_linear = non_linear
        if not non_linear:
            assert(dim1 == dim2)
            self.fc = nn.Linear(dim1, 1)
            torch.nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, x1, x2):
        z_walk = None
        if self.non_linear:
            x = torch.cat([x1, x2], dim=-1)
            h = self.act(self.fc1(x))
            z = self.fc2(h)
        else:
            x = torch.cat([x1, x2], dim=-2)
            z_walk = self.fc(x).squeeze(-1)
            z = z_walk.sum(dim=-1, keepdim=True)
        return z, z_walk


class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()
        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)

    def forward(self, ts):
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb


class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim

    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2)
        hn = seq_x.mean(dim=1)
        output = self.merger(hn, src_x)
        return output, None


class GRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_xn = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        r = torch.sigmoid(self.lin_xr(x) + self.lin_hr(h))
        z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h))
        g = torch.tanh(self.lin_xn(x) + self.lin_hn(r * h))
        return z * h + (1 - z) * g


class GRUODECell(torch.nn.Module):
    def __init__(self, hidden_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.bias = bias

        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t, h):
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        g = torch.tanh(x + self.lin_hn(r * h))
        dh = (1 - z) * (g - h)
        return dh


class NeurTWs(torch.nn.Module):
    def __init__(self, n_feat, e_feat, walk_mutual=False, walk_linear_out=False, pos_enc='saw', pos_dim=0, num_layers=3,
                 num_neighbors=20, tau=0.1, negs=1, solver='rk4', step_size=0.125, drop_out=0.1, cpu_cores=1,
                 verbosity=1, get_checkpoint_path=None):

        super(NeurTWs, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.verbosity = verbosity

        self.num_neighbors, self.num_layers = process_sampling_numbers(num_neighbors, num_layers)
        self.ngh_finder = None

        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)), requires_grad=False)
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)), requires_grad=False)

        self.feat_dim = self.n_feat_th.shape[1]
        self.e_feat_dim = self.e_feat_th.shape[1]
        self.time_dim = self.feat_dim
        self.pos_dim = pos_dim
        self.pos_enc = pos_enc
        self.model_dim = self.feat_dim + self.e_feat_dim + self.pos_dim
        self.logger.info('neighbors: {}, node dim: {}, edge dim: {}, pos dim: {}'.format(self.num_neighbors,
                                                                                         self.feat_dim,
                                                                                         self.e_feat_dim,
                                                                                         self.pos_dim))

        self.walk_mutual = walk_mutual
        self.walk_linear_out = walk_linear_out

        self.dropout_p = drop_out

        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        self.position_encoder = PositionEncoder(enc_dim=self.pos_dim, num_layers=self.num_layers,
                                                ngh_finder=self.ngh_finder, cpu_cores=cpu_cores,
                                                verbosity=verbosity, logger=self.logger, enc=self.pos_enc)

        self.solver = solver
        self.step_size = step_size

        self.walk_encoder = self.init_walk_encoder()
        self.logger.info('Encoding module - solver: {}, step size: {}'.format(self.solver, self.step_size))

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1, non_linear=not self.walk_linear_out)

        self.get_checkpoint_path = get_checkpoint_path

        self.flag_for_cur_edge = True
        self.common_node_percentages = {'pos': [], 'neg': []}
        self.walk_encodings_scores = {'encodings': [], 'scores': []}

        self.tau = tau
        self.negatives = negs
        self.f = lambda x: torch.exp(x / self.tau)
        self.logger.info('Contrastive module - temperature: {}, negatives: {}'.format(self.tau, self.negatives))

    def init_walk_encoder(self):
        walk_encoder = WalkEncoder(feat_dim=self.model_dim, pos_dim=self.pos_dim,
                                   model_dim=self.model_dim, out_dim=self.feat_dim,
                                   mutual=self.walk_mutual, dropout_p=self.dropout_p,
                                   logger=self.logger, walk_linear_out=self.walk_linear_out,
                                   solver=self.solver, step_size=self.step_size)
        return walk_encoder

    def contrast(self, src_idx_l, tgt_idx_l, bgd_idx_l, cut_time_l, e_idx_l=None):
        scores = 0.0

        subgraph_src = self.grab_subgraph(src_idx_l, cut_time_l, e_idx_l=e_idx_l)
        subgraph_tgt = self.grab_subgraph(tgt_idx_l, cut_time_l, e_idx_l=e_idx_l)

        self.position_encoder.init_internal_data(src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt)

        subgraph_src = self.subgraph_tree2walk(src_idx_l, cut_time_l, subgraph_src)
        subgraph_tgt = self.subgraph_tree2walk(tgt_idx_l, cut_time_l, subgraph_tgt)

        src_embed = self.forward_msg(src_idx_l, cut_time_l, subgraph_src)
        tgt_embed = self.forward_msg(tgt_idx_l, cut_time_l, subgraph_tgt)

        if self.walk_mutual:
            src_embed, tgt_embed = self.tune_msg(src_embed, tgt_embed)

        pos_score, _ = self.affinity_score(src_embed, tgt_embed)
        pos_score.squeeze(dim=-1)
        pos_score = self.f(pos_score.sigmoid())
        scores += pos_score
        size = len(src_idx_l)
        start_idx = 0

        for i in range(self.negatives):
            bgd_idx_cut = bgd_idx_l[start_idx : (i+1) * size]
            subgraph_src = self.grab_subgraph(src_idx_l, cut_time_l, e_idx_l=e_idx_l)
            subgraph_bgd = self.grab_subgraph(bgd_idx_cut, cut_time_l, e_idx_l=None)

            self.position_encoder.init_internal_data(src_idx_l, bgd_idx_cut, cut_time_l, subgraph_src, subgraph_bgd)
            subgraph_bgd = self.subgraph_tree2walk(bgd_idx_cut, cut_time_l, subgraph_bgd)
            bgd_embed = self.forward_msg(bgd_idx_cut, cut_time_l, subgraph_bgd)

            if self.walk_mutual:
                src_embed, bgd_embed = self.tune_msg(src_embed, bgd_embed)

            neg_score, _ = self.affinity_score(src_embed, bgd_embed)
            neg_score = self.f(neg_score.sigmoid())
            neg_score.squeeze(dim=-1)
            scores += neg_score
            start_idx = (i+1) * size

        loss = -torch.log(pos_score / scores)
        return loss.mean()

    def inference(self, src_idx_l, tgt_idx_l, bgd_idx_l, cut_time_l, e_idx_l=None):
        subgraph_src = self.grab_subgraph(src_idx_l, cut_time_l, e_idx_l=e_idx_l)
        subgraph_tgt = self.grab_subgraph(tgt_idx_l, cut_time_l, e_idx_l=e_idx_l)
        subgraph_bgd = self.grab_subgraph(bgd_idx_l, cut_time_l, e_idx_l=None)

        self.position_encoder.init_internal_data(src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt)

        subgraph_src = self.subgraph_tree2walk(src_idx_l, cut_time_l, subgraph_src)
        subgraph_tgt = self.subgraph_tree2walk(tgt_idx_l, cut_time_l, subgraph_tgt)

        src_embed = self.forward_msg(src_idx_l, cut_time_l, subgraph_src)
        tgt_embed = self.forward_msg(tgt_idx_l, cut_time_l, subgraph_tgt)

        if self.walk_mutual:
            src_embed, tgt_embed = self.tune_msg(src_embed, tgt_embed)

        pos_score, _ = self.affinity_score(src_embed, tgt_embed)
        pos_score.squeeze(dim=-1)
        pos_score = pos_score.sigmoid()

        subgraph_src = self.grab_subgraph(src_idx_l, cut_time_l, e_idx_l=e_idx_l)

        self.position_encoder.init_internal_data(src_idx_l, bgd_idx_l, cut_time_l, subgraph_src, subgraph_bgd)

        subgraph_bgd = self.subgraph_tree2walk(bgd_idx_l, cut_time_l, subgraph_bgd)
        bgd_embed = self.forward_msg(bgd_idx_l, cut_time_l, subgraph_bgd)

        if self.walk_mutual:
            src_embed, bgd_embed = self.tune_msg(src_embed, bgd_embed)

        neg_score, _ = self.affinity_score(src_embed, bgd_embed)
        neg_score.squeeze(dim=-1)
        neg_score = neg_score.sigmoid()
        return pos_score, neg_score

    def grab_subgraph(self, src_idx_l, cut_time_l, e_idx_l=None):
        subgraph = self.ngh_finder.find_k_hop(self.num_layers, src_idx_l, cut_time_l, num_neighbors=self.num_neighbors, e_idx_l=e_idx_l)
        return subgraph

    def subgraph_tree2walk(self, src_idx_l, cut_time_l, subgraph_src):
        node_records, eidx_records, t_records = subgraph_src
        node_records_tmp = [np.expand_dims(src_idx_l, 1)] + node_records
        eidx_records_tmp = [np.zeros_like(node_records_tmp[0])] + eidx_records
        t_records_tmp = [np.expand_dims(cut_time_l, 1)] + t_records

        new_node_records = self.subgraph_tree2walk_one_component(node_records_tmp)
        new_eidx_records = self.subgraph_tree2walk_one_component(eidx_records_tmp)
        new_t_records = self.subgraph_tree2walk_one_component(t_records_tmp)
        return new_node_records, new_eidx_records, new_t_records

    def subgraph_tree2walk_one_component(self, record_list):
        batch, n_walks, walk_len, dtype = record_list[0].shape[0], record_list[-1].shape[-1], len(record_list), record_list[0].dtype
        record_matrix = np.empty((batch, n_walks, walk_len), dtype=dtype)
        for hop_idx, hop_record in enumerate(record_list):
            assert(n_walks % hop_record.shape[-1] == 0)
            record_matrix[:, :, hop_idx] = np.repeat(hop_record, repeats=n_walks // hop_record.shape[-1], axis=1)
        return record_matrix

    def forward_msg(self, src_idx_l, cut_time_l, subgraph_src):
        node_records, eidx_records, t_records = subgraph_src
        hidden_embeddings, masks = self.init_hidden_embeddings(src_idx_l, node_records)
        edge_features = self.retrieve_edge_features(eidx_records)
        position_features = self.retrieve_position_features(src_idx_l, node_records, cut_time_l, t_records)

        t_records_th = torch.from_numpy(t_records).float().to(self.n_feat_th.device)
        final_node_embeddings = self.forward_msg_walk(hidden_embeddings, edge_features,
                                                      position_features, t_records_th, None)
        return final_node_embeddings

    def tune_msg(self, src_embed, tgt_embed):
        return self.walk_encoder.mutual_query(src_embed, tgt_embed)

    def init_hidden_embeddings(self, src_idx_l, node_records):
        device = self.n_feat_th.device
        node_records_th = torch.from_numpy(node_records).long().to(device)
        hidden_embeddings = self.node_raw_embed(node_records_th)
        masks = (node_records_th != 0).sum(dim=-1).long()
        return hidden_embeddings, masks

    def retrieve_edge_features(self, eidx_records):
        device = self.n_feat_th.device
        eidx_records_th = torch.from_numpy(eidx_records).to(device)
        eidx_records_th[:, :, 0] = 0
        edge_features = self.edge_raw_embed(eidx_records_th)
        return edge_features

    def retrieve_position_features(self, src_idx_l, node_records, cut_time_l, t_records):
        start = time.time()
        encode = self.position_encoder

        if encode.enc_dim == 0:
            return None

        batch, n_walk, len_walk = node_records.shape
        node_records_r, t_records_r = node_records.reshape(batch, -1), t_records.reshape(batch, -1)
        position_features, common_nodes, walk_encodings = encode(node_records_r, t_records_r)
        position_features = position_features.view(batch, n_walk, len_walk, self.pos_dim)
        self.update_common_node_percentages(common_nodes)
        end = time.time()

        if self.verbosity > 1:
            self.logger.info('encode positions encodings for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))

        return position_features

    def forward_msg_walk(self, hidden_embeddings, edge_features, position_features, t_records_th, masks):
        return self.walk_encoder.forward_one_node(hidden_embeddings, edge_features, position_features,
                                                  t_records_th, masks)

    def update_ngh_finder(self, ngh_finder):
        self.ngh_finder = ngh_finder
        self.position_encoder.ngh_finder = ngh_finder

    def update_common_node_percentages(self, common_node_percentage):
        if self.flag_for_cur_edge:
            self.common_node_percentages['pos'].append(common_node_percentage)
        else:
            self.common_node_percentages['neg'].append(common_node_percentage)

    def save_common_node_percentages(self, dir):
        torch.save(self.common_node_percentages, dir + '/common_node_percentages.pt')

    def save_walk_encodings_scores(self, dir):
        torch.save(self.walk_encodings_scores, dir + '/walk_encodings_scores.pt')


class PositionEncoder(nn.Module):
    def __init__(self, num_layers, enc='saw', enc_dim=2, ngh_finder=None, verbosity=1, cpu_cores=1, logger=None):
        super(PositionEncoder, self).__init__()
        self.enc = enc
        self.enc_dim = enc_dim
        self.num_layers = num_layers
        self.nodetime2emb_maps = None
        self.projection = nn.Linear(1, 1)
        self.cpu_cores = cpu_cores
        self.ngh_finder = ngh_finder
        self.verbosity = verbosity
        self.logger = logger

        if self.enc == 'spd':
            self.trainable_embedding = nn.Embedding(num_embeddings=self.num_layers+2, embedding_dim=self.enc_dim)
        else:
            assert(self.enc in ['lp', 'saw'])
            self.trainable_embedding = nn.Sequential(nn.Linear(in_features=self.num_layers+1, out_features=self.enc_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=self.enc_dim, out_features=self.enc_dim))

        self.logger.info("Distance encoding: {}".format(self.enc))

    def init_internal_data(self, src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt):
        if self.enc_dim == 0:
            return
        start = time.time()
        self.nodetime2emb_maps = self.collect_pos_mapping_ptree(src_idx_l, tgt_idx_l, cut_time_l,
                                                                subgraph_src, subgraph_tgt)
        end = time.time()
        if self.verbosity > 1:
            self.logger.info('init positions encodings for the minibatch, '
                             'time eclipsed: {} seconds'.format(str(end-start)))

    def collect_pos_mapping_ptree(self, src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt):
        if self.cpu_cores == 1:
            subgraph_src_node, _, subgraph_src_ts = subgraph_src
            subgraph_tgt_node, _, subgraph_tgt_ts = subgraph_tgt
            nodetime2emb_maps = {}
            for row in range(len(src_idx_l)):
                src = src_idx_l[row]
                tgt = tgt_idx_l[row]
                cut_time = cut_time_l[row]
                src_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_node]
                src_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_ts]
                tgt_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_node]
                tgt_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_ts]
                nodetime2emb_map = PositionEncoder.collect_pos_mapping_ptree_sample(src, tgt, cut_time,
                                                                                    src_neighbors_node,
                                                                                    src_neighbors_ts,
                                                                                    tgt_neighbors_node,
                                                                                    tgt_neighbors_ts,
                                                                                    batch_idx=row, enc=self.enc)
                nodetime2emb_maps.update(nodetime2emb_map)
        else:
            nodetime2emb_maps = {}
            cores = self.cpu_cores
            if cores in [-1, 0]:
                cores = mp.cpu_count()
            pool = mp.Pool(processes=cores)
            nodetime2emb_map = pool.map(PositionEncoder.collect_pos_mapping_ptree_sample_mp,
                                        [(src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt, row, self.enc)
                                         for row in range(len(src_idx_l))], chunksize=len(src_idx_l)//cores+1)
            pool.close()
            for i in range(len(nodetime2emb_map)):
                nodetime2emb_maps.update(nodetime2emb_map[i])

        return nodetime2emb_maps

    @staticmethod
    def collect_pos_mapping_ptree_sample(src, tgt, cut_time, src_neighbors_node, src_neighbors_ts,
                                         tgt_neighbors_node, tgt_neighbors_ts, batch_idx, enc='spd'):
        n_hop = len(src_neighbors_node)
        makekey = nodets2key
        nodetime2emb = {}

        if enc == 'spd':
            for k in range(n_hop-1, -1, -1):
                for src_node, src_ts, tgt_node, tgt_ts in zip(src_neighbors_node[k], src_neighbors_ts[k],
                                                              tgt_neighbors_node[k], tgt_neighbors_ts[k]):
                    src_key, tgt_key = makekey(batch_idx, src_node, src_ts), makekey(batch_idx, tgt_node, tgt_ts)
                    if src_key not in nodetime2emb:
                        nodetime2emb[src_key] = [k+1, 2*n_hop]
                    else:
                        nodetime2emb[src_key][0] = k+1
                    if tgt_key not in nodetime2emb:
                        nodetime2emb[tgt_key] = [2*n_hop, k+1]
                    else:
                        nodetime2emb[tgt_key][1] = k+1

            src_key = makekey(batch_idx, src, cut_time)
            tgt_key = makekey(batch_idx, tgt, cut_time)
            if src_key in nodetime2emb:
                nodetime2emb[src_key][0] = 0
            else:
                nodetime2emb[src_key] = [0, 2*n_hop]
            if tgt_key in nodetime2emb:
                nodetime2emb[tgt_key][1] = 0
            else:
                nodetime2emb[tgt_key] = [2*n_hop, 0]
            null_key = makekey(batch_idx, 0, 0.0)
            nodetime2emb[null_key] = [2 * n_hop, 2 * n_hop]

        elif enc == 'lp':
            src_neighbors_node, src_neighbors_ts = [[src]] + src_neighbors_node, [[cut_time]] + src_neighbors_ts
            tgt_neighbors_node, tgt_neighbors_ts = [[tgt]] + tgt_neighbors_node, [[cut_time]] + tgt_neighbors_ts
            for k in range(n_hop+1):
                k_hop_total = len(src_neighbors_node[k])
                for src_node, src_ts, tgt_node, tgt_ts in zip(src_neighbors_node[k], src_neighbors_ts[k],
                                                              tgt_neighbors_node[k], tgt_neighbors_ts[k]):
                    src_key, tgt_key = makekey(batch_idx, src_node, src_ts), makekey(batch_idx, tgt_node, tgt_ts)
                    if src_key not in nodetime2emb:
                        nodetime2emb[src_key] = np.zeros((2, n_hop+1), dtype=np.float32)
                    if tgt_key not in nodetime2emb:
                        nodetime2emb[tgt_key] = np.zeros((2, n_hop+1), dtype=np.float32)
                    nodetime2emb[src_key][0, k] += 1/k_hop_total
                    nodetime2emb[tgt_key][1, k] += 1/k_hop_total
            null_key = makekey(batch_idx, 0, 0.0)
            nodetime2emb[null_key] = np.zeros((2, n_hop + 1), dtype=np.float32)
            # nodetime2emb[(0, PositionEncoder.float2str(0.0))] = np.zeros((2, n_hop+1), dtype=np.float32)

        else:
            assert(enc == 'saw')
            src_neighbors_node, src_neighbors_ts = [[src]] + src_neighbors_node, [[cut_time]] + src_neighbors_ts
            tgt_neighbors_node, tgt_neighbors_ts = [[tgt]] + tgt_neighbors_node, [[cut_time]] + tgt_neighbors_ts
            src_seen_nodes2label = {}
            tgt_seen_nodes2label = {}
            for k in range(n_hop + 1):
                for src_node, src_ts, tgt_node, tgt_ts in zip(src_neighbors_node[k], src_neighbors_ts[k],
                                                              tgt_neighbors_node[k], tgt_neighbors_ts[k]):
                    src_key, tgt_key = makekey(batch_idx, src_node, src_ts), makekey(batch_idx, tgt_node, tgt_ts)

                    if src_key not in nodetime2emb:
                        nodetime2emb[src_key] = np.zeros((n_hop + 1, ), dtype=np.float32)
                    if src_node not in src_seen_nodes2label:
                        new_src_node_label = k
                        src_seen_nodes2label[src_key] = k
                    else:
                        new_src_node_label = src_seen_nodes2label[src_node]
                    nodetime2emb[src_key][new_src_node_label] = 1

                    if tgt_key not in nodetime2emb:
                        nodetime2emb[tgt_key] = np.zeros((n_hop + 1, ), dtype=np.float32)
                    if tgt_node not in tgt_seen_nodes2label:
                        new_tgt_node_label = k
                        tgt_seen_nodes2label[tgt_node] = k
                    else:
                        new_tgt_node_label = tgt_seen_nodes2label[tgt_node]
                    nodetime2emb[src_key][new_tgt_node_label] = 1
            null_key = makekey(batch_idx, 0, 0.0)
            nodetime2emb[null_key] = np.zeros((n_hop + 1, ), dtype=np.float32)

        return nodetime2emb

    def forward(self, node_record, t_record):
        device = next(self.projection.parameters()).device
        batched_keys = make_batched_keys(node_record, t_record)
        unique, inv = np.unique(batched_keys, return_inverse=True)
        unordered_encodings = np.array([self.nodetime2emb_maps[key] for key in unique])
        encodings = unordered_encodings[inv, :]
        encodings = torch.tensor(encodings).to(device)
        walk_encodings = None
        common_nodes = (((encodings.sum(-1) > 0).sum(-1) == 2).sum().float() / (encodings.shape[0] * encodings.shape[1])).item()
        encodings = self.get_trainable_encodings(encodings)
        return encodings, common_nodes, walk_encodings

    @staticmethod
    def collect_pos_mapping_ptree_sample_mp(args):
        src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt, row, enc = args
        subgraph_src_node, _, subgraph_src_ts = subgraph_src
        subgraph_tgt_node, _, subgraph_tgt_ts = subgraph_tgt
        src = src_idx_l[row]
        tgt = tgt_idx_l[row]
        cut_time = cut_time_l[row]
        src_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_node]
        src_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_ts]
        tgt_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_node]
        tgt_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_ts]
        nodetime2emb_map = PositionEncoder.collect_pos_mapping_ptree_sample(src, tgt, cut_time,
                                                                            src_neighbors_node, src_neighbors_ts,
                                                                            tgt_neighbors_node, tgt_neighbors_ts,
                                                                            batch_idx=row, enc=enc)
        return nodetime2emb_map

    def get_trainable_encodings(self, encodings):
        if self.enc == 'spd':
            encodings[encodings > (self.num_layers+0.5)] = self.num_layers + 1
            encodings = self.trainable_embedding(encodings.long())
            encodings = encodings.sum(dim=-2)
        elif self.enc == 'lp':
            encodings = self.trainable_embedding(encodings.float())
            encodings = encodings.sum(dim=-2)
        else:
            assert(self.enc == 'saw')
            encodings = self.trainable_embedding(encodings.float())
        return encodings


class WalkEncoder(nn.Module):
    def __init__(self, feat_dim, pos_dim, model_dim, out_dim, logger, mutual=False, dropout_p=0.1,
                 walk_linear_out=False, solver='rk4', step_size=0.125):

        super(WalkEncoder, self).__init__()

        self.solver = solver
        self.step_size = step_size
        self.feat_dim = feat_dim
        self.pos_dim = pos_dim
        self.model_dim = model_dim
        self.attn_dim = self.model_dim//2
        self.n_head = 8
        self.out_dim = out_dim
        self.mutual = mutual
        self.dropout_p = dropout_p
        self.logger = logger

        self.feature_encoder = FeatureEncoder(self.feat_dim, self.model_dim, self.dropout_p,
                                              self.solver, self.step_size)
        self.position_encoder = FeatureEncoder(self.pos_dim, self.pos_dim, self.dropout_p,
                                               self.solver, self.step_size)
        self.projector = nn.Sequential(nn.Linear(self.feature_encoder.hidden_dim + self.position_encoder.hidden_dim,
                                                 self.attn_dim), nn.ReLU(), nn.Dropout(self.dropout_p))

        if self.mutual:
            self.mutual_attention_src2tgt = TransformerDecoderLayer(d_model=self.attn_dim, nhead=self.n_head,
                                                                    dim_feedforward=4*self.model_dim,
                                                                    dropout=self.dropout_p,
                                                                    activation='relu')
            self.mutual_attention_tgt2src = TransformerDecoderLayer(d_model=self.attn_dim, nhead=self.n_head,
                                                                    dim_feedforward=4*self.model_dim,
                                                                    dropout=self.dropout_p,
                                                                    activation='relu')

        self.pooler = SetPooler(n_features=self.attn_dim, out_features=self.out_dim, dropout_p=self.dropout_p,
                                walk_linear_out=walk_linear_out)

    def forward_one_node(self, hidden_embeddings, edge_features, position_features, t_records, masks=None):
        combined_features = self.aggregate(hidden_embeddings, edge_features, position_features)
        combined_features = self.feature_encoder.integrate(t_records, combined_features, masks)
        if self.pos_dim > 0:
            position_features = self.position_encoder.integrate(t_records, position_features, masks)
            combined_features = torch.cat([combined_features, position_features], dim=-1)
        x = self.projector(combined_features)
        x = self.pooler(x, agg='mean')
        return x

    def mutual_query(self, src_embed, tgt_embed):
        src_emb = self.mutual_attention_src2tgt(src_embed, tgt_embed)
        tgt_emb = self.mutual_attention_tgt2src(tgt_embed, src_embed)
        src_emb = self.pooler(src_emb)
        tgt_emb = self.pooler(tgt_emb)
        return src_emb, tgt_emb

    def aggregate(self, hidden_embeddings, edge_features, position_features):
        batch, n_walk, len_walk, _ = hidden_embeddings.shape
        device = hidden_embeddings.device
        if position_features is None:
            assert(self.pos_dim == 0)
            combined_features = torch.cat([hidden_embeddings, edge_features], dim=-1)
        else:
            combined_features = torch.cat([hidden_embeddings, edge_features, position_features], dim=-1)
        combined_features = combined_features.to(device)
        assert(combined_features.size(-1) == self.feat_dim)
        return combined_features


class FeatureEncoder(nn.Module):
    start_time = 0.0
    end_time = 1.0

    def __init__(self, in_features, hidden_features, dropout_p=0.1, solver='rk4', step_size=0.125):
        super(FeatureEncoder, self).__init__()
        self.hidden_dim = hidden_features
        if self.hidden_dim == 0:
            return
        self.gru = GRUCell(in_features, hidden_features)
        self.odefun = GRUODECell(hidden_features)
        self.dropout = nn.Dropout(dropout_p)
        self.solver = solver
        if self.solver == 'euler' or self.solver == 'rk4':
            self.step_size = step_size

    def integrate(self, t_records, X, mask=None):
        batch, n_walk, len_walk, feat_dim = X.shape
        X = X.view(batch * n_walk, len_walk, feat_dim)
        t_records = t_records.view(batch * n_walk, len_walk, 1)
        h = torch.zeros(batch * n_walk, self.hidden_dim).type_as(X)
        for i in range(X.shape[1] - 1):
            h = self.gru(X[:, i, :], h)
            t0 = t_records[:, i+1, :]
            t1 = t_records[:, i, :]
            delta_t = torch.log10(torch.abs(t1 - t0) + 1.0) + 0.01
            h = (torch.zeros_like(t0), delta_t, h)
            if self.solver == 'euler' or self.solver == 'rk4':
                solution = odeint(self,
                                  h,
                                  torch.tensor([self.start_time, self.end_time]).type_as(X),
                                  method=self.solver,
                                  options=dict(step_size=self.step_size))
            elif self.solver == 'dopri5':
                solution = odeint(self,
                                  h,
                                  torch.tensor([self.start_time, self.end_time]).type_as(X),
                                  method=self.solver)
            else:
                raise NotImplementedError('{} solver is not implemented.'.format(self.solver))
            _, _, h = tuple(s[-1] for s in solution)
        encoded_features = self.gru(X[:, -1, :], h)
        encoded_features = encoded_features.view(batch, n_walk, self.hidden_dim)
        encoded_features = self.dropout(encoded_features)
        return encoded_features

    def forward(self, s, state):
        t0, t1, x = state
        ratio = (t1 - t0) / (self.end_time - self.start_time)
        t = (s - self.start_time) * ratio + t0
        dx = self.odefun(t, x)
        dx = dx * ratio
        return torch.zeros_like(t0), torch.zeros_like(t1), dx


class SetPooler(nn.Module):
    def __init__(self, n_features, out_features, dropout_p=0.1, walk_linear_out=False):
        super(SetPooler, self).__init__()
        self.mean_proj = nn.Linear(n_features, n_features)
        self.max_proj = nn.Linear(n_features, n_features)
        self.attn_weight_mat = nn.Parameter(torch.zeros((2, n_features, n_features)), requires_grad=True)
        nn.init.xavier_uniform_(self.attn_weight_mat.data[0])
        nn.init.xavier_uniform_(self.attn_weight_mat.data[1])
        self.dropout = nn.Dropout(dropout_p)
        self.out_proj = nn.Sequential(nn.Linear(n_features, out_features), nn.ReLU(), self.dropout)
        self.walk_linear_out = walk_linear_out

    def forward(self, X, agg='sum'):
        if self.walk_linear_out:
            return self.out_proj(X)
        if agg == 'sum':
            return self.out_proj(X.sum(dim=-2))
        else:
            assert(agg == 'mean')
            return self.out_proj(X.mean(dim=-2))


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):

        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_t = tgt.transpose(0, 1)
        tgt2 = self.self_attn(tgt_t, tgt_t, tgt_t, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation need be relu or gelu, not %s." % activation)