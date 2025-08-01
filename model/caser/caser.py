import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_, xavier_normal_, constant_

class Caser(nn.Module):
    r"""Caser is a model that incorporate CNN for recommendation.
    Note:
        We did not use the sliding window to generate training instances as in the paper, in order that
        the generation method we used is common to other sequential models.
        For comparison with other models, we set the parameter T in the paper as 1.
        In addition, to prevent excessive CNN layers (ValueError: Training loss is nan), please make sure the parameters MAX_ITEM_LIST_LENGTH small, such as 10.
    """

    def __init__(self, config):
        super(Caser, self).__init__()

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.n_h = config["n_h"]
        self.n_v = config["n_v"]
        self.dropout_prob = config["dropout_prob"]
        self.reg_weight = config["reg_weight"]
        self.n_items = config['n_items']
        self.max_seq_length = config['max_seq_length']

        # load dataset info
        # self.n_users = config['n_users']

        # define layers and loss
        # self.user_embedding = nn.Embedding(
        #     self.n_users, self.embedding_size, padding_idx=0
        # )
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )

       # vertical conv layer
        self.conv_v = nn.Conv2d(
            in_channels=1, out_channels=self.n_v, kernel_size=(self.max_seq_length, 1)
        )

        # horizontal conv layer
        lengths = [i + 1 for i in range(self.max_seq_length)]
        self.conv_h = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=self.n_h,
                    kernel_size=(i, self.embedding_size),
                )
                for i in lengths
            ]
        )

        # fully-connected layer
        self.fc1_dim_v = self.n_v * self.embedding_size
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, self.embedding_size)
        # self.fc2 = nn.Linear(
        #     self.embedding_size + self.embedding_size, self.embedding_size
        # )

        self.dropout = nn.Dropout(self.dropout_prob)
        self.ac_conv = nn.ReLU()
        self.ac_fc = nn.ReLU()

        self.loss_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 1.0 / module.embedding_dim)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, item_seq):
        # Embedding Look-up
        # use unsqueeze() to get a 4-D input for convolution layers. (batch_size * 1 * max_length * embedding_size)
        item_seq_emb = self.item_embedding(item_seq).unsqueeze(1)
        # user_emb = self.user_embedding(user).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_seq_emb)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect
        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)
        # fully-connected layer
        seq_output = self.ac_fc(self.fc1(out))
        # x = torch.cat([z, user_emb], 1)
        # seq_output = self.ac_fc(self.fc2(x))
        # the hidden_state of the predicted item, size:(batch_size * hidden_size)
        return seq_output

    def reg_loss_conv_h(self):
        r"""
        L2 loss on conv_h
        """
        loss_conv_h = 0
        for name, parm in self.conv_h.named_parameters():
            if name.endswith("weight"):
                loss_conv_h = loss_conv_h + parm.norm(2)
        return self.reg_weight * loss_conv_h


    def calculate_loss(self, user, item_seq, time_seq, item_seq_len, pos_items):
        seq_output = self.forward(item_seq)
        test_item_emb = self.item_embedding.weight # [item_num H]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)
        return loss

    # def predict(self, interaction):
    #     item_seq = interaction[self.ITEM_SEQ]
    #     test_item = interaction[self.ITEM_ID]
    #     seq_output = self.forward(item_seq)
    #     test_item_emb = self.item_embedding(test_item)
    #     scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
    #     return scores

    def full_sort_predict(self, user, item_seq, time_seq, item_seq_len):
        # item_seq = interaction[self.ITEM_SEQ]
        # item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq)
        test_item_emb = self.item_embedding.weight  # [item_num H]
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B, item_num]
        return scores
