import torch
from torch import nn
from entmax import entmax_bisect
from rotary_embedding_torch import RotaryEmbedding


# 前馈神经网络，用于增加网络的非线性处理能力
# 使得 Transformer 不仅仅依赖于注意力机制中的 加权求和 来处理信息
class FF(nn.Module):
    """
    Feed-forward in a transformer layer.
    """
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.lin_1 = nn.Linear(input_size, hidden_size)
        self.lin_2 = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()
        self.droupout = nn.Dropout(dropout)

    def forward(self, x):
        # 首先经过一个线形层，然后经过一个激活函数，最后经过一个dropout层
        output = self.lin_2(self.droupout(self.relu(self.droupout(self.lin_1(x)))))
        return self.droupout(output)

# 多头注意力
class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention block in a transformer layer.
    """
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        # 确保可以整除
        assert dim % self.n_heads == 0
        self.att_size = int(dim / n_heads)

        # 生成Q,K,V
        self._query = nn.Linear(dim, dim, bias=False)
        self._key = nn.Linear(dim, dim, bias=False)
        self._value = nn.Linear(dim, dim, bias=False)

        # Attention Block
        self.dense = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dim, dim, bias=False)

        self.rotary_emb = RotaryEmbedding(dim=64)

    def forward(self, q, k, v, time_lengths, mask=None):
        scale_factor = torch.sqrt(torch.FloatTensor([self.att_size])).item()
        batch_size = q.size(0)

        # To Multiple Attention Heads
        # 形状 (batch_size, seq_len, feature_dim)->(batch_size, seq_len, n_heads, att_size)->(batch_size, n_heads, seq_len, att_size)
        _query = self._query(q).view(batch_size, -1, self.n_heads, self.att_size).transpose(1, 2)
        _key = self._key(k).view(batch_size, -1, self.n_heads, self.att_size).transpose(1, 2)
        _value = self._value(v).view(batch_size, -1, self.n_heads, self.att_size).transpose(1, 2)
        # Scaled dot-product Attention score
        # score.shape (batch_size, n_heads, seq_len, seq_len)
        # time_lengths.shape (batch_size, seq_len)
        # 分开操作，都转换为0到1之间以后再加权一起

        score = torch.matmul(_query, _key.transpose(-2, -1)) / scale_factor

        # # #
        # if time_lengths is not None:
        #     leng = time_lengths.size(1)
        #     time_weights = time_lengths.unsqueeze(1).unsqueeze(2).repeat(1, 1, leng, 1)
        #     score = time_weights + score

        if mask is not None:
            score = score + mask

        # # 对缩放后的权重进行 softmax 归一化，得到注意力权重
        score = self.softmax(score)

        # 将注意力权重应用于 V 向量，计算加权和，得到加权信息
        z = torch.matmul(self.dropout(score), _value)
        # 将所有头的结果拼接起来
        z = z.transpose(1, 2).reshape(batch_size, -1, self.att_size * self.n_heads)
        # 对拼接后的结果进行线性变化
        z = self.dense(z)
        return self.dropout(z)

class EncoderCell(nn.Module):
    """
    Encoder Cell contains MultiHeadAttention > Add & LayerNorm1 > Feed Forward > Add & LayerNorm2
    """

    def __init__(self, input_size, hidden_size, n_heads , dropout):
        super().__init__()
        # Attention Block
        self.mh_attention = MultiHeadAttention(input_size, n_heads, dropout)
        self.lnorm_1 = nn.LayerNorm(input_size)
        # Feed forward block
        self.ff = FF(input_size, hidden_size, dropout)
        self.lnorm_2 = nn.LayerNorm(input_size)

    def forward(self, q, k, v, time_lengths, mask):
        # add and norm 先残差连接再层归一化
        x = self.mh_attention(q, k, v, time_lengths, mask)
        x = self.lnorm_1(x)+ q
        x = self.lnorm_2(self.ff(x) + x)
        # x = self.mh_attention(self.lnorm_1(q), k, v, time_lengths, mask) + q
        # x = self.ff(self.lnorm_2(x)) + x
        return x

class Encoder(nn.Module):
    """
    Encoder Block with n stacked encoder cells.
    """
    def __init__(self, input_size, hidden_size: int = 2048, n_layers: int = 1 , n_heads: int = 1 , dropout: float = 0.1):
        super().__init__()
        # Stack of encoder-cells n_layers high
        self.stack = nn.ModuleList()
        # Building encoder stack
        for layer in range(n_layers):
            self.stack.append(EncoderCell(input_size, hidden_size, n_heads, dropout))
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, time_lengths, mask):
        x = q
        for cell in self.stack:
            x = cell(self.dropout(x), self.dropout(k), self.dropout(v),time_lengths, mask)
        return x


# class EncoderCell(nn.Module):
#     """
#     Encoder Cell contains MultiHeadAttention > Add & LayerNorm1 > Feed Forward > Add & LayerNorm2
#     """
#     def __init__(self, input_size, hidden_size, n_heads , dropout):
#         super().__init__()
#         # Attention Block
#         self.mh_attention = MultiHeadAttention(input_size, n_heads, dropout)
#         self.lnorm_1 = nn.LayerNorm(input_size)
#         # Feed forward block
#         self.ff = FF(input_size, hidden_size, dropout)
#         self.lnorm_2 = nn.LayerNorm(input_size)
#
#     def forward(self, x, time_lengths, mask):
#         # add and norm 先残差连接再层归一化
#         x = self.lnorm_1(self.mh_attention(x, x, x, time_lengths, mask)+ x)
#         x = self.lnorm_2(self.ff(x) + x)
#         return x
#
# class Encoder(nn.Module):
#     """
#     Encoder Block with n stacked encoder cells.
#     """
#     def __init__(self, input_size, hidden_size: int = 2048, n_layers: int = 1 , n_heads: int = 1 , dropout: float = 0.1):
#         super().__init__()
#         # Stack of encoder-cells n_layers high
#         self.stack = nn.ModuleList()
#         # Building encoder stack
#         for layer in range(n_layers):
#             self.stack.append(EncoderCell(input_size, hidden_size, n_heads, dropout))
#         # Dropout layer
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, time_lengths, mask):
#         for cell in self.stack:
#             x = cell(self.dropout(x), time_lengths, mask)
#         return x

# 稀疏注意力版本
# import torch
# from torch import nn
# from entmax import entmax_bisect
#
#
#
# # 前馈神经网络，用于增加网络的非线性处理能力
# # 使得 Transformer 不仅仅依赖于注意力机制中的 加权求和 来处理信息
# class FF(nn.Module):
#     """
#     Feed-forward in a transformer layer.
#     """
#     def __init__(self, input_size, hidden_size, dropout):
#         super().__init__()
#         self.lin_1 = nn.Linear(input_size, hidden_size)
#         self.lin_2 = nn.Linear(hidden_size, input_size)
#         self.relu = nn.ReLU()
#         self.droupout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         # 首先经过一个线形层，然后经过一个激活函数，最后经过一个dropout层
#         output = self.lin_2(self.droupout(self.relu(self.droupout(self.lin_1(x)))))
#         return self.droupout(output)
#
# # 多头注意力
# class MultiHeadAttention(nn.Module):
#     """
#     Multi-head Attention block in a transformer layer.
#     """
#     def __init__(self, dim, n_heads, dropout):
#         super().__init__()
#         self.n_heads = n_heads
#         # 确保可以整除
#         assert dim % self.n_heads == 0
#         self.att_size = int(dim / n_heads)
#
#         # 生成Q,K,V
#         self._query = nn.Linear(dim, dim, bias=False)
#         self._key = nn.Linear(dim, dim, bias=False)
#         self._value = nn.Linear(dim, dim, bias=False)
#
#         # Attention Block
#         self.dense = nn.Linear(dim, dim, bias=False)
#         self.softmax = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)
#         self.out = nn.Linear(dim, dim, bias=False)
#
#     def forward(self, q, k, v, time_lengths, mask, alpha_ent = 1):
#         scale_factor = torch.sqrt(torch.FloatTensor([self.att_size])).item()
#         batch_size = q.size(0)
#
#         # To Multiple Attention Heads
#         # 形状 (batch_size, seq_len, feature_dim)->(batch_size, seq_len, n_heads, att_size)->(batch_size, n_heads, seq_len, att_size)
#         _query = self._query(q).view(batch_size, -1, self.n_heads, self.att_size).transpose(1, 2)
#         _key = self._key(k).view(batch_size, -1, self.n_heads, self.att_size).transpose(1, 2)
#         _value = self._value(v).view(batch_size, -1, self.n_heads, self.att_size).transpose(1, 2)
#         # Scaled dot-product Attention score
#         # score.shape (batch_size, n_heads, seq_len, seq_len)
#         # time_lengths.shape (batch_size, seq_len)
#         # 分开操作，都转换为0到1之间以后再加权一起
#         score = torch.matmul(_query, _key.transpose(-2, -1)) / scale_factor
#         # #
#         if time_lengths is not None:
#             leng = time_lengths.size(1)
#             time_weights = time_lengths.unsqueeze(1).unsqueeze(2).repeat(1, 1, leng, 1)
#             score = time_weights + score
#
#         if mask is not None:
#             score = score + mask
#         # # 对缩放后的权重进行 softmax 归一化，得到注意力权重
#         score = entmax_bisect(score, alpha_ent, dim=-1)
#
#         # 将注意力权重应用于 V 向量，计算加权和，得到加权信息
#         z = torch.matmul(self.dropout(score), _value)
#         # 将所有头的结果拼接起来
#         z = z.transpose(1, 2).reshape(batch_size, -1, self.att_size * self.n_heads)
#         # 对拼接后的结果进行线性变化
#         z = self.dense(z)
#         return self.dropout(z)
#
# class EncoderCell(nn.Module):
#     """
#     Encoder Cell contains MultiHeadAttention > Add & LayerNorm1 > Feed Forward > Add & LayerNorm2
#     """
#     def __init__(self, input_size, hidden_size, n_heads , dropout):
#         super().__init__()
#         # Attention Block
#         self.mh_attention = MultiHeadAttention(input_size, n_heads, dropout)
#         self.lnorm_1 = nn.LayerNorm(input_size)
#         # Feed forward block
#         self.ff = FF(input_size, hidden_size, dropout)
#         self.lnorm_2 = nn.LayerNorm(input_size)
#
#     def forward(self, x, time_lengths, mask, alpha_ent):
#         # add and norm 先残差连接再层归一化
#         x = self.lnorm_1(self.mh_attention(x, x, x, time_lengths, mask, alpha_ent)+ x)
#         x = self.lnorm_2(self.ff(x) + x)
#         return x
#
# class Encoder(nn.Module):
#     """
#     Encoder Block with n stacked encoder cells.
#     """
#     def __init__(self, input_size, hidden_size: int = 2048, n_layers: int = 1 , n_heads: int = 1 , dropout: float = 0.1):
#         super().__init__()
#         # Stack of encoder-cells n_layers high
#         self.stack = nn.ModuleList()
#         # Building encoder stack
#         for layer in range(n_layers):
#             self.stack.append(EncoderCell(input_size, hidden_size, n_heads, dropout))
#         # Dropout layer
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, time_lengths, mask , alpha_ent):
#         for cell in self.stack:
#             x = cell(self.dropout(x), time_lengths, mask, alpha_ent)
#         return x



