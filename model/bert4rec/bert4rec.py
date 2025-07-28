import random

import torch
from torch import nn

from recbole.model.layers import TransformerEncoder
from model.bert4rec.transformer import Encoder

class BERT4Rec(nn.Module):

    def __init__(self, config):
        super(BERT4Rec, self).__init__()

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.mask_ratio = config['mask_ratio']
        self.initializer_range = config['initializer_range']
        # dataset info
        self.n_items = config['n_items']
        self.max_seq_length = config['max_seq_length']
        
        # load dataset info
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)


        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # mask token add 1
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)  # add mask_token at the last

        # self.trm_encoder = TransformerEncoder(
        #     n_layers=self.n_layers,
        #     n_heads=self.n_heads,
        #     hidden_size=self.hidden_size,
        #     inner_size=self.inner_size,
        #     hidden_dropout_prob=self.hidden_dropout_prob,
        #     attn_dropout_prob=self.attn_dropout_prob,
        #     hidden_act=self.hidden_act,
        #     layer_norm_eps=self.layer_norm_eps
        # )

        # 定义一个TransformerEncoderLayer

        self.transfomerlayer = Encoder(self.hidden_size, self.inner_size, self.n_layers, self.n_heads,
                                       self.hidden_dropout_prob)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _neg_sample(self, item_set):
        item = random.randint(1, self.n_items - 1)
        while item in item_set:
            item = random.randint(1, self.n_items - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def reconstruct_train_data(self, item_seq):
        """
        Mask item sequence for training.
        """
        device = item_seq.device
        batch_size = item_seq.size(0)

        sequence_instances = item_seq.cpu().numpy().tolist()

        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        neg_items = []
        masked_index = []
        for instance in sequence_instances:
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            pos_item = []
            neg_item = []
            index_ids = []
            for index_id, item in enumerate(instance):
                # padding is 0, the sequence is end
                if item == 0:
                    break
                prob = random.random()
                if prob < self.mask_ratio:
                    pos_item.append(item)
                    neg_item.append(self._neg_sample(instance))
                    masked_sequence[index_id] = self.mask_token
                    index_ids.append(index_id)

            masked_item_sequence.append(masked_sequence)
            pos_items.append(self._padding_sequence(pos_item, self.mask_item_length))
            neg_items.append(self._padding_sequence(neg_item, self.mask_item_length))
            masked_index.append(self._padding_sequence(index_ids, self.mask_item_length))

        # [B Len]
        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(batch_size, -1)
        return masked_item_sequence, pos_items, neg_items, masked_index

    # 原始版本
    def reconstruct_test_data(self, item_seq, item_seq_len):
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        return item_seq
    #更改后
    def reconstruct_test_data(self, item_seq, item_seq_len, time_seq):
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        time_seq = torch.cat((time_seq, padding.unsqueeze(-1)), dim=-1)

        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
            time_seq[batch_id][last_position] = self.mask_token
        return item_seq, time_seq


    def forward(self, item_seq , time_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        #原始的实现方法
        # extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        # trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        # output = trm_output[-1]
        #第二种实现方法
        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        output = self.transfomerlayer(input_emb, input_emb, input_emb, time_seq, extended_attention_mask)
        return output  # [B L H]

    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.
        Examples:
            sequence: [1 2 3 4 5]
            masked_sequence: [1 mask 3 mask 5]
            masked_index: [1, 3]
            max_length: 5
            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_loss(self, user, item_seq, time_seq, item_seq_len, pos_items):
        masked_item_seq, pos_items, neg_items, masked_index = self.reconstruct_train_data(item_seq)

        seq_output = self.forward(masked_item_seq, time_seq)
        pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1))  # [B*mask_len max_len]
        # [B mask_len] -> [B mask_len max_len] multi hot
        pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1)  # [B mask_len max_len]
        # [B mask_len max_len] * [B max_len H] -> [B mask_len H]
        # only calculate loss for masked position
        seq_output = torch.bmm(pred_index_map, seq_output)  # [B mask_len H]

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        test_item_emb = self.item_embedding.weight[:self.n_items]  # [item_num H]

        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B mask_len item_num]
        targets = (masked_index > 0).float().view(-1)  # [B*mask_len]

        loss = torch.sum(loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1)) * targets) \
               / torch.sum(targets)
        return loss

    def predict(self, item_seq, item_seq_len, test_item):
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len)  # [B H]
        test_item_emb = self.item_embedding(test_item)
        # scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]

        # a 的每一行与 b 的每一行的转置进行矩阵乘法
        scores = torch.matmul(seq_output.unsqueeze(1), test_item_emb.transpose(2, 1))
        # 去掉不必要的维度
        scores = scores.squeeze(1)
        return scores

    def full_sort_predict(self, user, item_seq, time_seq, item_seq_len):
        item_seq, time_seq = self.reconstruct_test_data(item_seq, item_seq_len, time_seq)
        seq_output = self.forward(item_seq, time_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len)  # [B H]
        test_item_emb = self.item_embedding.weight[:self.n_items]  # [item_num H]
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B, item_num]
        return scores

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            # 生成下三角矩阵 下半部分为0
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
            # 将符合条件的位置替换为 0，不符合条件的位置替换为 -10000.0
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)

        return extended_attention_mask

    # def get_attention_mask(self, item_seq, bidirectional=False):
    #     """
    #     Generate left-to-right uni-directional or bidirectional attention mask,
    #     and apply masking for positions where item_seq == 0.
    #     返回形状为 [seq_len, seq_len] 的掩码.
    #     """
    #
    #     # 获取序列长度
    #     seq_len = item_seq.size(1)
    #     # 生成注意力掩码 (下三角掩码或全 1 矩阵)
    #     if bidirectional:
    #         # 双向注意力: 全 1 矩阵
    #         attention_mask = torch.ones((seq_len, seq_len), device=item_seq.device).bool()
    #     else:
    #         # 单向注意力: 下三角矩阵
    #         attention_mask = torch.tril(torch.ones((seq_len, seq_len), device=item_seq.device)).bool()
    #     # 填充掩码 (item_seq == 0 的位置)
    #     padding_mask = (item_seq != 0).unsqueeze(1).expand(-1, seq_len, -1).bool()
    #     # 结合填充掩码与注意力掩码
    #     combined_mask = attention_mask & padding_mask.any(dim=0)
    #     # 将符合条件的位置替换为 0，不符合条件的位置替换为 -10000.0
    #     combined_mask = torch.where(combined_mask, torch.tensor(0.0, device=item_seq.device), torch.tensor(-10000.0, device=item_seq.device))
    #
    #     return combined_mask
