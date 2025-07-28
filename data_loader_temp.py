import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
from torch import nn

def get_dataloader(config, d_type=None):

    dataset = SeqDataset(config["data_file"], config['max_seq_length'], type=d_type)
    if config['method'] == 'caser':
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle = True,
            drop_last = False,
            collate_fn = collate_fn_max_len,
            num_workers=config['num_workers'],
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle = True,
            drop_last = False,
            collate_fn = collate_fn,
            num_workers=config['num_workers'],
        )
    return dataloader

# this data loader split the dataset according to leave one out strategy.
class SeqDataset(Dataset):

    def __init__(self, file_name, max_seq_length, type=None):
        self.file_frame = pd.read_csv(file_name)
        self.type = type
        self.max_seq_length = max_seq_length

    def __getitem__(self, index):
        data = self.file_frame.iloc[index]
        input_ids = list(eval(data.book_ids))
        times = list(eval(data.times))

        # input_ids = list(eval(data.book_ids))
        # times = list(eval(data.play_time))

        input_ids = input_ids[-self.max_seq_length:]
        times = times[-self.max_seq_length:]

        item_seq = []
        time_seq = []
        pos_item = 0
        if self.type == 'train':
            item_seq = input_ids[:-3]
            pos_item = input_ids[-3]
            time_seq = times[:-3]
        elif self.type == 'val':
            item_seq = input_ids[:-2]
            pos_item = input_ids[-2]
            time_seq = times[:-2]
        elif self.type == 'test':
            item_seq = input_ids[:-1]
            pos_item = input_ids[-1]
            time_seq = times[:-1]

        user = data.user_id
        item_seq = torch.tensor(item_seq)
        time_seq = torch.tensor(time_seq)
        item_seq_len = len(item_seq)
        pos_item = pos_item

        rated = {pos_item}
        rated.add(0)

        cand_items = [pos_item]

        return user, item_seq, time_seq, item_seq_len, pos_item, cand_items

    def __len__(self):
        return len(self.file_frame)

def collate_fn(batch_data):
    ret = list()
    for idx, data_list in enumerate(zip(*batch_data)):
        if isinstance(data_list[0], torch.Tensor) and (idx==1 or idx==2):
            item_seq = rnn_utils.pad_sequence(data_list, batch_first=True, padding_value=0)
            ret.append(item_seq)
        else:
            ret.append(torch.tensor(data_list))
    return tuple(ret)

def collate_fn_max_len(batch_data):
    ret = list()
    for idx, data_list in enumerate(zip(*batch_data)):
        if isinstance(data_list[0], torch.Tensor) and (idx==1 or idx==2):
            # print(data_list[0])
            # print(nn.ConstantPad1d((0, 50-data_list[0].shape[0]), 0)(data_list[0]))
            data_list = list(data_list)
            data_list[0] = nn.ConstantPad1d((0, 100-data_list[0].shape[0]), 0)(data_list[0])
            data_list = tuple(data_list)
            # data_list.sort(key=lambda data: len(data), reverse=True)
            item_seq = rnn_utils.pad_sequence(data_list, batch_first=True, padding_value=0)
            # print(item_seq)
            ret.append(item_seq)
        else:
            ret.append(torch.tensor(data_list))
    # (user, item_seq, item_seq_length, pos_item, neg_item) --> tensor format
    return tuple(ret)