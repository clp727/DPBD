import torch
from torch import nn
import pandas as pd
from transformers import BertTokenizer, BertModel
from recbole.model.layers import TransformerEncoder

import sys
sys.path.append('..')




class SASRec(nn.Module):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.
    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config):
        super(SASRec, self).__init__()

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']

        # dataset info
        self.n_items = config['n_items']
        self.max_seq_length = config['max_seq_length']
        self.n_book_classification = config['n_book_classification']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.loss_fct = nn.CrossEntropyLoss()


        # 书籍类别
        self.embeddings_classification = nn.Embedding(
            self.n_book_classification + 1, self.hidden_size
        )
        # 图书id和图书类别的映射
        self.book_to_category = pd.read_csv("/home/chenliping/skjj_all/data/book.csv", index_col='book_id')['classification'].to_dict()

        # 图书id和图书名的映射 一种是放到提取偏好里面 一种是直接作为图书特征
        self.book_to_title = pd.read_csv("/home/chenliping/skjj_all/data/book.csv", index_col='book_id')['title'].to_dict()

        # 预训练BERT模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained("/home/chenliping/skjj_all/bert/bert_base_chinese/")
        self.bert_model = BertModel.from_pretrained("/home/chenliping/skjj_all/bert/bert_base_chinese/")

        self.linear_layer = nn.Linear(768, self.hidden_size)
        self.fusion_layer = DimAttention(self.hidden_size, self.hidden_size)
        # parameters initialization
        self.apply(self._init_weights)

    # 对模型中的每个子模块进行权重初始化，以便网络能够从合理的参数开始训练。
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

    def get_classification_embeddings(self, books_id):
        categories = [self.book_to_category.get(book_id.item(), 0) for book_id in books_id.view(-1)]
        category_tensor = torch.tensor(categories, dtype=torch.long, device= books_id.device).view(books_id.shape)
        classification_emb = self.embeddings_classification(category_tensor)
        return classification_emb


    def get_booktitle_embeddings(self, books_id):
        titles = [self.book_to_title.get(book_id.item(), "0") for book_id in books_id.view(-1)]
        inputs = self.tokenizer(titles, return_tensors='pt', padding=True, truncation=True, max_length=30)
        inputs = {key: value.to(books_id.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        # 把每个词的向量平均起来[batch_size, sequence_length, hidden_size]-》[batch_size, hidden_size]
        title_embeddings = outputs.last_hidden_state.mean(dim=1)
        title_embeddings = title_embeddings.view(books_id.shape[0], books_id.shape[1], -1)
        title_embeddings = self.linear_layer(title_embeddings)
        return title_embeddings


    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        # 标记序列中哪些位置是有效的（非填充的)
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            # 生成一个下三角矩阵掩码（包括对角线），保留对角线及其以下的位置，掩盖对角线以上的部分
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, torch.tensor(0.0, device=extended_attention_mask.device), torch.tensor(-10000.0, device=extended_attention_mask.device))
        return extended_attention_mask


    def encode(self, item_seq, position_embedding):
        books_emb = self.item_embedding(item_seq)
        classification_emb = self.get_classification_embeddings(item_seq)
        title_emb = self.get_booktitle_embeddings(item_seq)
        features = torch.cat((books_emb,classification_emb,title_emb,position_embedding), dim = 2)
        features = self.fusion_layer(features)
        return features

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        # input_emb = self.encode(item_seq,position_embedding)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]

        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, user, item_seq, time_seq, item_seq_len, pos_items):
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding.weight  # [item_num H]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)
        return loss
    # def predict(self, item_seq, item_seq_len, test_item):
    #     seq_output = self.forward(item_seq, item_seq_len)
    #     test_item_emb = self.item_embedding(test_item)
    #     scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
    #     return scores

    # def predict(self, item_seq, item_seq_len, test_item):
    #     seq_output = self.forward(item_seq, item_seq_len)
    #     test_item_emb = self.item_embedding(test_item)
    #     # a 的每一行与 b 的每一行的转置进行矩阵乘法
    #     scores = torch.matmul(seq_output.unsqueeze(1), test_item_emb.transpose(2, 1))
    #     # 去掉不必要的维度
    #     scores = scores.squeeze(1)
    #     return scores

    def full_sort_predict(self, user, item_seq, time_seq, item_seq_len):
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding.weight  # [item_num H]
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B n_items]
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
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask


# class DimAttention(nn.Module):
#     """
#     Attention layer to convert [batch_size, seq_len, dim*3] -> [batch_size, seq_len, dim]
#     """
#     def __init__(self, hidden_dim, attn_dim):
#         super(DimAttention, self).__init__()
#         # 定义投影层，用于计算注意力得分
#         self.projection = nn.Sequential(
#             nn.Linear(hidden_dim * 4, attn_dim),  # hidden_dim * 3 是拼接后的特征维度
#             nn.ReLU(True),
#             nn.Linear(attn_dim, 4)  # 输出 3 个权重值（分别对应 dim, dim, dim）
#         )
#         self.softmax = nn.Softmax(dim=-1)  # 归一化到 [0, 1]
#
#     def forward(self, input_tensor):
#         """
#         Args:
#             input_tensor: 输入张量, 形状 [batch_size, seq_len, dim*4]
#
#         Returns:
#             hidden_states: 输出融合后的张量, 形状 [batch_size, seq_len, dim]
#         """
#         batch_size, seq_len, _ = input_tensor.shape
#         # 将 input_tensor 变形为 [batch_size, seq_len, 4, dim]
#         input_tensor_reshaped = input_tensor.view(batch_size, seq_len, 4, -1)  # 分成 4 个 [batch_size, seq_len, dim] 张量
#
#         # 使用投影层计算注意力得分 [batch_size, seq_len, 4]
#         energy = self.projection(input_tensor)
#
#         # 计算注意力权重 [batch_size, seq_len, 4]
#         attention_weights = self.softmax(energy)
#
#         # 加权求和生成 [batch_size, seq_len, dim]
#         hidden_states = (input_tensor_reshaped * attention_weights.unsqueeze(-1)).sum(dim=2)
#
#         return hidden_states

class DimAttention(nn.Module):
    """
    Attention layer to convert [batch_size, seq_len, 4, dim] -> [batch_size, seq_len, 4, 1]
    """
    def __init__(self, hidden_dim, attn_dim):
        super(DimAttention, self).__init__()
        # 定义投影层，用于计算注意力得分
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),  # hidden_dim * 3 是拼接后的特征维度
            nn.ReLU(True),
            nn.Linear(attn_dim, 1)  # 输出 1 个权重值
        )
        self.softmax = nn.Softmax(dim=-1)  # 归一化到 [0, 1]

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: 输入张量, 形状 [batch_size, seq_len, dim*4]

        Returns:
            hidden_states: 输出融合后的张量, 形状 [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = input_tensor.shape
        # 将 input_tensor 变形为 [batch_size, seq_len, 4, dim]
        input_tensor_reshaped = input_tensor.view(batch_size, seq_len, 4, -1)  # 分成 4 个 [batch_size, seq_len, dim] 张量
        # [batch_size, seq_len, 4, 1]
        energy = self.projection(input_tensor_reshaped)

        # 计算注意力权重  [batch_size, seq_len, 4, 1] ->[batch_size, seq_len, 4] 去除最后一个维度以后，在四个特征上softmax
        attention_weights = self.softmax(energy.squeeze(-1))

        # 加权求和生成 [batch_size, seq_len, 4, dim]
        hidden_states = (input_tensor_reshaped * attention_weights.unsqueeze(-1)).sum(dim=2)

        return hidden_states
