import torch
import torch.nn as nn
from modules.base_model import BaseModel
from utils.parse_args import args
import torch.nn.functional as F
from modules.utils import EdgelistDrop
import numpy as np
import scipy.sparse as sp
import math
import networkx as nx
import random
from copy import deepcopy
# from torch_scatter import scatter_add, scatter_sum
from modules.utils import scatter_add, scatter_sum
from torch.nn import GRUCell

init = nn.init.xavier_uniform_


class evolveGCN_O(BaseModel):
    def __init__(self, dataset, pretrained_model=None, last_emb=None):
        super().__init__(dataset)
        self.adj = self._make_binorm_adj(dataset.graph)
        self.edges = self.adj._indices().t()
        self.edge_norm = self.adj._values()

        self.gru_cell = GRUCell(input_size=self.emb_size, hidden_size=self.emb_size)
        
        if pretrained_model is not None:
            pre_user_emb, pre_item_emb = pretrained_model.generate()
            self.user_embedding = nn.Parameter(pre_user_emb).requires_grad_(True)
            self.item_embedding = nn.Parameter(pre_item_emb).requires_grad_(True)
        elif last_emb is not None:
            last_user_emb, last_item_emb = last_emb.split([self.num_users, self.num_items], dim=0)
            self.user_embedding = nn.Parameter(last_user_emb).requires_grad_(True)
            self.item_embedding = nn.Parameter(last_item_emb).requires_grad_(True)
        else:
            self.user_embedding = nn.Parameter(init(torch.empty(self.num_users, self.emb_size)))
            self.item_embedding = nn.Parameter(init(torch.empty(self.num_items, self.emb_size)))
            print("using xavier init...")

        self.edge_dropout = EdgelistDrop()

    def _agg(self, all_emb, edges, edge_norm):
        src_emb = all_emb[edges[:, 0]]

        # bi-norm
        src_emb = src_emb * edge_norm.unsqueeze(1)

        # conv
        dst_emb = scatter_sum(src_emb, edges[:, 1], dim=0, dim_size=self.num_users+self.num_items)
        return dst_emb
    
    def _edge_binorm(self, edges):
        user_degs = scatter_add(torch.ones_like(edges[:, 0]), edges[:, 0], dim=0, dim_size=self.num_users)
        user_degs = user_degs[edges[:, 0]]
        item_degs = scatter_add(torch.ones_like(edges[:, 1]), edges[:, 1], dim=0, dim_size=self.num_items)
        item_degs = item_degs[edges[:, 1]]
        norm = torch.pow(user_degs, -0.5) * torch.pow(item_degs, -0.5)
        return norm

    def forward(self, edges, edge_norm):
        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        all_emb = self.gru_cell(all_emb, all_emb)
        res_emb = [all_emb]
        for l in range(args.num_layers):
            all_emb = self._agg(res_emb[-1], edges, edge_norm)
            res_emb.append(all_emb)
        res_emb = sum(res_emb)
        user_res_emb, item_res_emb = res_emb.split([self.num_users, self.num_items], dim=0)
        return user_res_emb, item_res_emb
    
    def cal_loss(self, batch_data):
        edges, dropout_mask = self.edge_dropout(self.edges, 1-args.edge_dropout, return_mask=True)
        edge_norm = self.edge_norm[dropout_mask]

        # forward
        users, pos_items, neg_items = batch_data
        user_emb, item_emb = self.forward(edges, edge_norm)
        batch_user_emb = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        neg_item_emb = item_emb[neg_items]
        rec_loss = self._bpr_loss(batch_user_emb, pos_item_emb, neg_item_emb)
        reg_loss = args.weight_decay * self._reg_loss(users, pos_items, neg_items)

        loss = rec_loss + reg_loss
        loss_dict = {
            "rec_loss": rec_loss.item(),
            "reg_loss": reg_loss.item(),
        }
        return loss, loss_dict
    
    @torch.no_grad()
    def generate(self):
        return self.forward(self.edges, self.edge_norm)
    
    @torch.no_grad()
    def rating(self, user_emb, item_emb):
        return torch.matmul(user_emb, item_emb.t())
    
    def _reg_loss(self, users, pos_items, neg_items):
        u_emb = self.user_embedding[users]
        pos_i_emb = self.item_embedding[pos_items]
        neg_i_emb = self.item_embedding[neg_items]
        reg_loss = (1/2)*(u_emb.norm(2).pow(2) +
                          pos_i_emb.norm(2).pow(2) +
                          neg_i_emb.norm(2).pow(2))/float(len(users))
        return reg_loss
