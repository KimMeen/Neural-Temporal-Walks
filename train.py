import torch
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from eval import *
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True


def train_val(train_val_data, model, mode, bs, epochs, optimizer, early_stopper, ngh_finders,
              rand_samplers, logger, negatives):
    train_data, val_data = train_val_data
    train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = train_data
    val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = val_data
    train_rand_sampler, val_rand_sampler = rand_samplers
    partial_ngh_finder, full_ngh_finder = ngh_finders
    if mode == 't':
        model.update_ngh_finder(full_ngh_finder)
    elif mode == 'i':
        model.update_ngh_finder(partial_ngh_finder)
    else:
        raise ValueError('training mode {} not found.'.format(mode))
    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / bs)
    idx_list = np.arange(num_instance)
    for epoch in range(epochs):
        ap, auc, m_loss = [], [], []
        np.random.shuffle(idx_list)
        logger.info('start {} epoch'.format(epoch))
        for k in tqdm(range(num_batch)):
            s_idx = k * bs
            e_idx = min(num_instance - 1, s_idx + bs)
            if s_idx == e_idx:
                continue
            batch_idx = idx_list[s_idx:e_idx]
            src_l_cut, dst_l_cut = train_src_l[batch_idx], train_dst_l[batch_idx]
            ts_l_cut = train_ts_l[batch_idx]
            e_l_cut = train_e_idx_l[batch_idx]
            label_l_cut = train_label_l[batch_idx]

            size = len(src_l_cut)
            _, dst_l_fake = train_rand_sampler.sample(negatives * size)
            optimizer.zero_grad()
            model.train()
            loss = model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                size = len(src_l_cut)
                _, dst_l_fake = train_rand_sampler.sample(size)
                pos_prob, neg_prob = model.inference(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut)
                pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                ap.append(average_precision_score(true_label, pred_score))
                m_loss.append(loss.item())
                auc.append(roc_auc_score(true_label, pred_score))

        val_ap, val_auc = eval_one_epoch(model, val_rand_sampler, val_src_l, val_dst_l,
                                         val_ts_l, val_label_l, val_e_idx_l)
        logger.info('epoch: {}:'.format(epoch))
        logger.info('epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('train auc: {}, val auc: {}'.format(np.mean(auc), val_auc))
        logger.info('train ap: {}, val ap: {}'.format(np.mean(ap), val_ap))

        if epoch == 0:
            checkpoint_dir = '/'.join(model.get_checkpoint_path(0).split('/')[:-1])
            # model.ngh_finder.save_ngh_stats(checkpoint_dir)
            model.save_common_node_percentages(checkpoint_dir)

        if early_stopper.early_stop_check(val_ap):
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_checkpoint_path = model.get_checkpoint_path(early_stopper.best_epoch)
            model.load_state_dict(torch.load(best_checkpoint_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        else:
            torch.save(model.state_dict(), model.get_checkpoint_path(epoch))


