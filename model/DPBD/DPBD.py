import math
import pandas as pd
import torch.nn as nn
import torch
import torch.multiprocessing
import os
from transformers import BertTokenizer, BertModel
import sys

sys.path.append('..')
from transformers import BertTokenizer, BertModel
from model.DPBD.transformer import Encoder
import torch.nn.functional as F

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class TIME_MODEL(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.dropout_prob = config['dropout_prob']

        # dataset info
        self.n_items = config['n_items']
        self.n_users = config['n_users']
        self.n_book_classification = config['n_book_classification']
        self.max_seq_length = config['max_seq_length']
        self.initializer_range = config['initializer_range']
        self.layer_norm_eps = config['layer_norm_eps']

        # Layer definitions
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.dropout_prob)

        # Embedding layers
        self.embeddings_user_id = nn.Embedding(self.n_users + 1, self.hidden_size)
        self.embeddings_book_id = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.embeddings_classification = nn.Embedding(self.n_book_classification + 1, self.hidden_size)

        # Book metadata mappings
        self.book_to_category = pd.read_csv("./data/book.csv", index_col='book_id')['classification'].to_dict()
        self.book_to_title = pd.read_csv("./data/book.csv", index_col='book_id')['title'].to_dict()

        # BERT for text features
        self.tokenizer = BertTokenizer.from_pretrained("./bert/bert_base_chinese/")
        self.bert_model = BertModel.from_pretrained("./bert/bert_base_chinese/")
        self.linear_layer = nn.Linear(768, self.hidden_size)

        # Position embeddings
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        # Dual-path transformers
        self.item_transformer = Encoder(self.hidden_size, self.inner_size,
                                        self.n_layers, self.n_heads, self.dropout_prob)
        self.feature_transformer = Encoder(self.hidden_size, self.inner_size,
                                           self.n_layers, self.n_heads, self.dropout_prob)

        # Fusion components
        self.fuser = GatedFusion(self.hidden_size)
        self.fuse_proj = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=True)
        self.ac_fc = nn.ReLU()

        # Loss function
        self.loss_fct = nn.CrossEntropyLoss()

    def encode_input(self, item_seq):
        """Feature-level path: encode book features"""
        classification_emb = self.get_classification_embeddings(item_seq)
        booktitle_emb = self.get_booktitle_embeddings(item_seq)
        feats = torch.stack([classification_emb, booktitle_emb], dim=2)  # [B, L, 2, D]
        fused_feats = self.fuser(feats)
        return fused_feats

    def get_classification_embeddings(self, books_id):
        categories = [self.book_to_category.get(book_id.item(), 0) for book_id in books_id.view(-1)]
        category_tensor = torch.tensor(categories, dtype=torch.long, device=books_id.device).view(books_id.shape)
        return self.embeddings_classification(category_tensor)

    def get_booktitle_embeddings(self, books_id):
        titles = [self.book_to_title.get(book_id.item(), "0") for book_id in books_id.view(-1)]
        inputs = self.tokenizer(titles, return_tensors='pt', padding=True, truncation=True, max_length=30)
        inputs = {key: value.to(books_id.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        title_embeddings = outputs.last_hidden_state.mean(dim=1)
        title_embeddings = title_embeddings.view(books_id.shape[0], books_id.shape[1], -1)
        return self.linear_layer(title_embeddings)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate attention mask for transformers"""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask,
                                              torch.tensor(0.0, device=extended_attention_mask.device),
                                              torch.tensor(-10000.0, device=extended_attention_mask.device))
        return extended_attention_mask

    def normalize_time_scores(self, time_seq):
        """Normalize borrowing time scores as described in the paper"""
        # Step 1: Filter out invalid times (too short or too long)
        # These thresholds should come from config or be calculated from data
        mu_a = 1  # minimum effective borrowing duration
        mu_b = 30  # maximum valid borrowing duration (example value)

        # Apply thresholds
        t_prime = torch.where((time_seq <= mu_a) | (time_seq >= mu_b),
                              torch.zeros_like(time_seq),
                              time_seq)

        # Step 2: Normalize by user's average borrowing time
        user_avg_time = time_seq.mean(dim=1, keepdim=True)  # [B, 1]
        t_double_prime = t_prime / (user_avg_time + 1e-8)  # [B, L]

        # Step 3: Min-max normalization to [0, 1]
        min_t = t_double_prime.min(dim=1, keepdim=True)[0]
        max_t = t_double_prime.max(dim=1, keepdim=True)[0]
        time_scores = (t_double_prime - min_t) / (max_t - min_t + 1e-8)

        return time_scores

    def forward(self, user, item_seq, time_seq, item_seq_len):
        """Dual-path forward pass as described in the paper"""
        # ========== Item-level path ==========
        # Get item embeddings and position embeddings
        books_emb = self.embeddings_book_id(item_seq)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # Normalize time scores
        time_scores = self.normalize_time_scores(time_seq) if time_seq is not None else None

        # Item-level transformer with time-aware attention
        item_features = books_emb + position_embedding
        mask = self.get_attention_mask(item_seq)
        item_output = self.item_transformer(item_features, item_features, books_emb, time_scores, mask)
        item_output = self.gather_indexes(item_output, item_seq_len - 1)  # [B, D]

        # ========== Feature-level path ==========
        # Get feature embeddings
        feature_emb = self.encode_input(item_seq)  # [B, L, D]

        # Feature-level transformer (no time information)
        feature_output = self.feature_transformer(feature_emb, feature_emb, feature_emb, None, mask)
        feature_output = self.gather_indexes(feature_output, item_seq_len - 1)  # [B, D]

        # ========== Fusion ==========
        # Concatenate and project as per paper
        fused_output = torch.cat([item_output, feature_output], dim=-1)  # [B, 2D]
        final_output = self.fuse_proj(fused_output)  # [B, D]

        return final_output

    def calculate_loss(self, user, item_seq, time_seq, item_seq_len, pos_items):
        seq_output = self.forward(user, item_seq, time_seq, item_seq_len)
        test_item_emb = self.embeddings_book_id.weight  # [item_num H]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, user, item_seq, time_seq, item_seq_len):
        seq_output = self.forward(user, item_seq, time_seq, item_seq_len)
        test_item_emb = self.embeddings_book_id.weight  # [item_num H]
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B n_items]
        return scores

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)


class GatedFusion(nn.Module):
    """Gated fusion mechanism for feature-level path"""

    def __init__(self, hidden_size):
        super().__init__()
        self.gate_fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, feats):
        """
        feats: [B, L, K=2, D]  (classification_emb and title_emb)
        """
        cls_emb = feats[:, :, 0, :]  # [B, L, D]
        title_emb = feats[:, :, 1, :]  # [B, L, D]

        # Compute gate
        cat = torch.cat([cls_emb, title_emb], dim=-1)  # [B, L, 2D]
        g = torch.sigmoid(self.gate_fc(cat))  # [B, L, 1]

        # Gated fusion
        fused = g * cls_emb + (1 - g) * title_emb  # [B, L, D]
        return fused