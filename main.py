import pandas as pd
from log import *
from utils import *
from train import *
from module import NeurTWs
from graph import NeighborFinder
import resource

args, sys_argv = get_args()

assert(args.cpu_cores >= -1)
set_random_seed(args.seed)

device = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available() else torch.device('cpu')

g_df = pd.read_csv('./data/ml_{}.csv'.format(args.data))
if args.data_usage < 1:
    g_df = g_df.iloc[:int(args.data_usage*g_df.shape[0])]
    print('use partial data, ratio: {}'.format(args.data_usage), flush=True)
e_feat = np.load('./data/ml_{}.npy'.format(args.data))
n_feat = np.load('./data/ml_{}_node.npy'.format(args.data))
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values
max_idx = max(src_l.max(), dst_l.max())

assert(np.unique(np.stack([src_l, dst_l])).shape[0] == max_idx or ~math.isclose(1, args.data_usage))
assert(n_feat.shape[0] == max_idx + 1 or ~math.isclose(1, args.data_usage))

val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv)

if args.mode == 't':
    logger.info('Transductive training...')
    valid_train_flag = (ts_l <= val_time)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time
else:
    assert(args.mode == 'i')
    logger.info('Inductive training...')
    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
    num_total_unique_nodes = len(total_node_set)
    mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])),
                                      int(0.1 * num_total_unique_nodes)))  # mask 10% nodes for inductive evaluation
    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_mask_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
    valid_train_flag = (ts_l <= val_time) * (none_mask_node_flag > 0.5)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time) * (none_mask_node_flag > 0.5)
    valid_test_flag = (ts_l > test_time) * (none_mask_node_flag < 0.5)
    valid_test_new_new_flag = (ts_l > test_time) * mask_src_flag * mask_dst_flag
    valid_test_new_old_flag = (valid_test_flag.astype(int) - valid_test_new_new_flag.astype(int)).astype(bool)
    logger.info('Sampled {} nodes (10 %) which are masked in training and reserved for testing...'.format(
        len(mask_node_set)))

train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = \
    src_l[valid_train_flag], dst_l[valid_train_flag], ts_l[valid_train_flag], \
    e_idx_l[valid_train_flag], label_l[valid_train_flag]

val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = \
    src_l[valid_val_flag], dst_l[valid_val_flag], ts_l[valid_val_flag], \
    e_idx_l[valid_val_flag], label_l[valid_val_flag]

test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = \
    src_l[valid_test_flag], dst_l[valid_test_flag], ts_l[valid_test_flag], \
    e_idx_l[valid_test_flag], label_l[valid_test_flag]

if args.mode == 'i':
    test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_e_idx_new_new_l, test_label_new_new_l = \
        src_l[valid_test_new_new_flag], dst_l[valid_test_new_new_flag], ts_l[valid_test_new_new_flag], \
        e_idx_l[valid_test_new_new_flag], label_l[valid_test_new_new_flag]

    test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_e_idx_new_old_l, test_label_new_old_l = \
        src_l[valid_test_new_old_flag], dst_l[valid_test_new_old_flag], ts_l[valid_test_new_old_flag], \
        e_idx_l[valid_test_new_old_flag], label_l[valid_test_new_old_flag]

train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l
val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l
train_val_data = (train_data, val_data)

full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))

full_ngh_finder = NeighborFinder(full_adj_list, temporal_bias=args.temporal_bias, spatial_bias=args.spatial_bias,
                                 ee_bias=args.ee_bias, use_cache=args.ngh_cache, sample_method=args.pos_sample,
                                 limit_ngh_span=args.limit_ngh_span, ngh_span=args.ngh_span)

partial_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
for src, dst, eidx, ts in zip(val_src_l, val_dst_l, val_e_idx_l, val_ts_l):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))

partial_ngh_finder = NeighborFinder(partial_adj_list, temporal_bias=args.temporal_bias, spatial_bias=args.spatial_bias,
                                    ee_bias=args.ee_bias, use_cache=args.ngh_cache, sample_method=args.pos_sample,
                                    limit_ngh_span=args.limit_ngh_span, ngh_span=args.ngh_span)

ngh_finders = partial_ngh_finder, full_ngh_finder
logger.info('Sampling module - temporal bias: {}, spatial bias: {}, E&E bias: {}'.format(args.temporal_bias,
                                                                                         args.spatial_bias,
                                                                                         args.ee_bias))

train_rand_sampler = RandEdgeSampler((train_src_l, ), (train_dst_l, ))
val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_dst_l, val_dst_l))
test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l))
rand_samplers = train_rand_sampler, val_rand_sampler

model = NeurTWs(n_feat=n_feat, e_feat=e_feat, walk_mutual=args.walk_mutual, walk_linear_out=args.walk_linear_out,
                pos_enc=args.pos_enc, pos_dim=args.pos_dim, num_layers=args.n_layer, num_neighbors=args.n_degree,
                tau=args.tau, negs=args.negs, solver=args.solver, step_size=args.step_size, drop_out=args.drop_out,
                cpu_cores=args.cpu_cores, verbosity=args.verbosity, get_checkpoint_path=get_checkpoint_path).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
early_stopper = EarlyStopMonitor(tolerance=args.tolerance)

train_val(train_val_data, model, args.mode, args.bs, args.n_epoch, optimizer, early_stopper,
          ngh_finders, rand_samplers, logger, args.negs)

model.update_ngh_finder(full_ngh_finder)
test_ap, test_auc = eval_one_epoch(model, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l)
logger.info('Test statistics: {} all nodes -- auc: {}, ap: {}'.format(args.mode, test_auc, test_ap))

test_new_new_ap, test_new_new_auc, test_new_old_ap, test_new_old_auc = [-1]*4
if args.mode == 'i':
    test_new_new_ap, test_new_new_auc = eval_one_epoch(model, test_rand_sampler, test_src_new_new_l,
                                                       test_dst_new_new_l, test_ts_new_new_l,
                                                       test_label_new_new_l, test_e_idx_new_new_l)
    logger.info('Test statistics: {} new-new nodes -- auc: {}, ap: {}'.format(args.mode, test_new_new_auc,
                                                                              test_new_new_ap))
    test_new_old_ap, test_new_old_auc = eval_one_epoch(model, test_rand_sampler, test_src_new_old_l,
                                                       test_dst_new_old_l, test_ts_new_old_l,
                                                       test_label_new_old_l, test_e_idx_new_old_l)
    logger.info('Test statistics: {} new-old nodes -- auc: {}, ap: {}'.format(args.mode, test_new_old_auc,
                                                                              test_new_old_ap))

logger.info('Saving model...')
torch.save(model.state_dict(), best_model_path)
logger.info('Saved model to {}'.format(best_model_path))
logger.info('model saved')