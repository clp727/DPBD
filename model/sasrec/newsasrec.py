import numpy as np
import torch
from torch import nn
import sys
import torch.nn.functional as F

sys.path.append('..')

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, config):
        super(SASRec, self).__init__()
        self.dev = config['device']
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_embedding = torch.nn.Embedding(config['n_items']+1, config['hidden_size'], padding_idx=0)
        self.pos_emb = torch.nn.Embedding(config['max_seq_length'] + 1, config['hidden_size'], padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=config['hidden_dropout_prob'])

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(config['hidden_size'], eps=1e-8)
        self.loss_fct = nn.CrossEntropyLoss()

        self.concat = AttentionMerge(config['hidden_size'])

        for _ in range(config['n_layers']):
            new_attn_layernorm = torch.nn.LayerNorm(config['hidden_size'], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(config['hidden_size'],
                                                          config['n_heads'],
                                                          config['attn_dropout_prob'])
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(config['hidden_size'], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(config['hidden_size'],config['hidden_dropout_prob'])
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, item_seq): # TODO: fp64 and int64 as default in python, trim?
        seqs = self.item_embedding(item_seq)

        seqs *= self.item_embedding.embedding_dim ** 0.5
        pre_seq = seqs
        poss = torch.arange(1, item_seq.shape[1] + 1, device=item_seq.device).unsqueeze(0).repeat(item_seq.shape[0], 1)
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (item_seq != 0)
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            pre_seq = torch.transpose(pre_seq, 0, 1)

            Q = self.attention_layernorms[i](seqs) #层归一化
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, item_seq, item_seq_len): # for training
        log_feats = self.log2feats(item_seq) # user_ids hasn't been used yet
        output = self.gather_indexes(log_feats, item_seq_len - 1)

        #output = self.concat(log_feats, item_seq_len - 1)
        att = None
        old_score = None
        return output,att,old_score# pos_pred, neg_pred

    def calculate_loss(self, user, item_seq, time_seq, item_seq_len, pos_items):
        seq_output,_,_ = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding.weight  # [item_num H]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)
        return loss,_,_

    def predict(self, item_seq, item_indices): # for inference
        log_feats = self.log2feats(item_seq) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)

    def full_sort_predict(self, user, item_seq, time_seq, item_seq_len):
        seq_output,_,_  = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding.weight  # [item_num H]
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B n_items]
        return scores,_,_

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)


class AttentionMerge(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionMerge, self).__init__()
        # 定义线性变换
        self.W1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W2 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.b1 = nn.Parameter(torch.zeros(embedding_dim))
        # 可以使用learnable query向量
        self.query = nn.Parameter(torch.randn(embedding_dim))

    def forward(self, transformer_output_p, item_seq_len):
        # transformer_output_p 是 (batch_size, seq_length, embedding_dim)
        batch_size, seq_length, embedding_dim = transformer_output_p.size()

        # 使用最后一个项目的向量作为 v_n
        v_n = transformer_output_p[torch.arange(batch_size), item_seq_len - 1]  # (batch_size, embedding_dim)

        # 计算注意力权重 a_i
        # W1 * v_n + W2 * v_i
        v_n_proj = self.W1(v_n).unsqueeze(1).expand_as(transformer_output_p)  # (batch_size, seq_length, embedding_dim)
        v_i_proj = self.W2(transformer_output_p)  # (batch_size, seq_length, embedding_dim)

        # 激活函数 sigma (这里使用 tanh, 你也可以尝试 ReLU 或者其他激活函数)
        scores = torch.tanh(v_n_proj + v_i_proj + self.b1)  # (batch_size, seq_length, embedding_dim)

        # 计算 a_i，query 向量与 score 进行内积，最后通过 softmax 得到注意力权重
        a_i = torch.matmul(scores, self.query)  # (batch_size, seq_length)
        a_i = F.softmax(a_i, dim=-1)  # 通过 softmax 归一化 (batch_size, seq_length)

        # 加权求和 Sn = sum(a_i * v_i)
        a_i = a_i.unsqueeze(-1)  # (batch_size, seq_length, 1)
        S_n = torch.sum(a_i * transformer_output_p, dim=1)  # (batch_size, embedding_dim)

        return S_n
