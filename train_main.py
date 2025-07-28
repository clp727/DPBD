import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm
import logging
from data_loader_temp import get_dataloader
from model.bert4rec.bert4rec import BERT4Rec
from model.caser.caser import Caser
from model.gru4rec.gru4rec import GRU4Rec
from model.sasrec.sasrec import SASRec
from model.srgnn.srgnn import SRGNN
from model.stamp.stamp import STAMP
from model.DPBD.DPBD import TIME_MODEL
from utils import convert_to_gpu, convert_all_data_to_gpu, get_local_time, set_logger, EarlyStopping
import metrics

max_seq_length = 10
def parse_args():
    parser = argparse.ArgumentParser()
    # basic args
    parser.add_argument("--n_items", default=37385, type=int)
    parser.add_argument("--max_seq_length", default=max_seq_length, type=int)

    # train args
    parser.add_argument("--batch_size", default=64, type=int, help="number of train_batch_size")
    parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument("--lr", default=0.001, type=float, help="learning rate of adam")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam second beta value")

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda:3", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_file", default = "./data/data.csv", type=str)

    parser.add_argument("--patience", default=10, type=int, help="how long to wait after last time validation loss improved")

    return parser.parse_args()


def get_config(model_name):
    args = parse_args()
    input_config = args.__dict__
    model_config_path = os.getcwd() + f"/model/{model_name}/model_config.yaml"
    with open(model_config_path, 'rb') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    model_config['method'] = model_name
    config = {**model_config, **input_config}

    return config


def create_model(config):
    if config['method'] == 'gru4rec':
        model  = GRU4Rec(config)
    elif config['method'] == 'sasrec':
        model = SASRec(config)
    elif config['method'] == 'bert4rec':
        model = BERT4Rec(config)
    elif config['method'] == 'caser':
        model = Caser(config)
    elif config['method'] == 'srgnn':
        model = SRGNN(config)
    elif config['method'] == 'stamp':
        model = STAMP(config)
    elif config['method'] == 'DPBD':
        model = TIME_MODEL(config)
    return model

def eval_acc(topk_list, predict_list, label_list):
    acc_dict = {acc+str(topk):[] for acc in ['recall', 'ndcg', 'mrr'] for topk in topk_list}
    for predict, label in zip(predict_list, label_list):
        for topk in topk_list:
            acc_dict['recall'+str(topk)].append(metrics.recall(predict, label, topk))
            acc_dict['ndcg'+str(topk)].append(metrics.ndcg(predict, label, topk))
            acc_dict['mrr'+str(topk)].append(metrics.mrr(predict, label, topk))
    avg_acc_dict = {metric: round(np.mean(result), 7) for metric, result in acc_dict.items()}
    return avg_acc_dict

def check_nan(loss):
    if torch.isnan(loss):
        raise ValueError('Training loss is nan')

def train_model(method):
    config = get_config(method)

    log_path = os.path.join(config['output_dir']+ config['method'] , get_local_time() + '.log')
    config['checkpoint_path'] = os.path.join(config['output_dir'] + config['method'] , get_local_time()+ '.pt')

    logger = set_logger(log_path)
    logger.info(str(config))

    train_data_loader, val_data_loader, test_data_loader = get_dataloader(config, d_type='train'), get_dataloader(config, d_type='val'), get_dataloader(config, d_type='test')

    model = create_model(config)
    model = convert_to_gpu(model, device=config['device'])
    logger.info(model)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 weight_decay=config['weight_decay'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)


    model.train()
    topk_list = [1, 5, 10]
    val_best_acc_dict = {acc+str(topk): 0 for acc in ['recall', 'ndcg', 'mrr'] for topk in topk_list}
    val_metric_anchor = 'recall10'

    early_stopping = EarlyStopping(config['checkpoint_path'], logger=logger, patience=config['patience'], verbose=True)

    for epoch in range(config['epochs']):
        tqdm_train_iter = tqdm(enumerate(train_data_loader),
                               desc="Mode_%s:%d" % ("train", epoch),
                               total=len(train_data_loader),
                               bar_format="{l_bar}{r_bar}")
        model.train()
        total_loss = 0
        for step, (user, item_seq, time_seq, item_seq_len, pos_items, cand_items) in tqdm_train_iter:
            user, item_seq, time_seq, item_seq_len, pos_items, cand_items  = convert_all_data_to_gpu(user,item_seq, time_seq, item_seq_len, pos_items, cand_items, device=config["device"])
            optimizer.zero_grad()
            loss= model.calculate_loss(user, item_seq, time_seq, item_seq_len, pos_items)
            total_loss += loss.cpu().data.numpy()
            check_nan(loss)
            loss.backward()
            optimizer.step()

        epoch_avg_loss = total_loss/ len(tqdm_train_iter)
        logger.info({"epoch":epoch, "train_loss:": epoch_avg_loss})

        # val eval
        tqdm_val_iter = tqdm(enumerate(val_data_loader),
                             desc="Mode_%s:%d" % ("validate", epoch),
                             total=len(val_data_loader),
                             bar_format="{l_bar}{r_bar}")
        model.eval()

        val_rec_ranks_list = []
        val_user_list = []
        val_pos_item_list = []
        val_item_seq_list = []

        total_val_loss = 0
        for step, (user, item_seq, time_seq, item_seq_len, pos_items, cand_items) in tqdm_val_iter:
            user, item_seq, time_seq, item_seq_len, pos_items, cand_items  = convert_all_data_to_gpu(user,item_seq,time_seq, item_seq_len, pos_items, cand_items, device=config["device"])
            val_loss= model.calculate_loss(user, item_seq, time_seq, item_seq_len, pos_items)
            total_val_loss += val_loss.cpu().data.numpy()
            predict= model.full_sort_predict(user, item_seq, time_seq, item_seq_len) # [b, n_items]
            predict = predict.cpu()
            _, rec_ranks = torch.topk(predict, 50, dim=-1) #save top 50
            val_rec_ranks_list.extend(rec_ranks.tolist())
            val_pos_item_list.extend(pos_items.cpu().data.tolist())


        epoch_val_loss = total_val_loss / len(tqdm_val_iter)
        logger.info({"epoch": epoch, "val_loss:": epoch_val_loss})

        val_acc_dict = eval_acc(topk_list, val_rec_ranks_list, val_pos_item_list)
        logger.info(f"Epoch: {epoch}, val_acc: {val_acc_dict}")

        tqdm_test_iter = tqdm(enumerate(test_data_loader),
                              desc="Mode_%s" % ("test"),
                              total=len(test_data_loader),
                              bar_format="{l_bar}{r_bar}")
        model.eval()
        test_rec_ranks_list = []
        test_pos_item_list = []
        for step, (user, item_seq, time_seq, item_seq_len, pos_items, cand_items)  in tqdm_test_iter:
            user, item_seq, time_seq, item_seq_len, pos_items, cand_items  = convert_all_data_to_gpu(user,item_seq,
                                                                                                     time_seq,
                                                                                                     item_seq_len,
                                                                                                     pos_items,
                                                                                                     cand_items,
                                                                                                     device=config["device"])

            predict = model.full_sort_predict(user, item_seq, time_seq, item_seq_len)  # [b, n_items]
            # predict = model.predict(item_seq, item_seq_len, cand_items)
            predict = predict.cpu()
            _, rec_ranks = torch.topk(predict, 50, dim=-1)  # save top 50

            test_rec_ranks_list.extend(rec_ranks.tolist())
            test_pos_item_list.extend(pos_items.cpu().data.tolist())

        test_acc_dict = eval_acc(topk_list, test_rec_ranks_list, test_pos_item_list)
        logger.info(f"Epoch: {epoch}, test_acc: {test_acc_dict}")

        # find best model epoch, save prediction, sasrec performance, and the model
        if val_acc_dict[val_metric_anchor] > val_best_acc_dict[val_metric_anchor]:
            val_best_acc_dict = val_acc_dict
            best_epoch = epoch
            overall = test_acc_dict

    logger.info(f"Epoch: {best_epoch}, test_acc: {overall}")

if __name__ == '__main__':
    # train_model('caser')
    # train_model('gru4rec')
    # train_model('sasrec')
    # train_model('srgnn')
    #train_model('stamp')
    train_model('DPBD')
    # train_model('bert4rec')
