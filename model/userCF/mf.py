import  torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class MF(nn.Module):
    def __init__(self,user_nums,item_nums,embedding_dim=64,opt="Adam",reg_user=1e-5,reg_item=1e-5):

        super(MF,self).__init__()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.items_emb=nn.Embedding(item_nums+1,embedding_dim,device=self.device)
        self.users_emb=nn.Embedding(user_nums+1,embedding_dim,device=self.device)
        self.reg_user=reg_user
        self.reg_item= reg_item
        self.loss_fn=F.cross_entropy
        self.total_loss=0
        self.opt=opt

    def forward(self,user,item):
        user_emb = self.users_emb(user)
        item_emb = self.items_emb(item)
        res =torch.dot(user_emb, item_emb)
        return res


    def calculate_loss(self, item_seq, item_seq_len, pos_items, neg_items=None):
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding.weight # [item_num H]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        # B*K B
        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, item_seq, item_seq_len):
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding.weight  # [item_num H]
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B, n_items]
        return scores