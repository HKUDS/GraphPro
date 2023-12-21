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

@torch.no_grad()
def average_state_dict(state_dict1: dict, state_dict2: dict, weight: float) -> dict:
    # Average two model.state_dict() objects.
    # out = (1-w)*dict1 + w*dict2
    assert 0 <= weight <= 1
    d1 = deepcopy(state_dict1)
    d2 = deepcopy(state_dict2)
    out = dict()
    for key in d1.keys():
        assert isinstance(d1[key], torch.Tensor)
        param1 = d1[key].detach().clone()
        assert isinstance(d2[key], torch.Tensor)
        param2 = d2[key].detach().clone()
        out[key] = (1 - weight) * param1 + weight * param2
    return out

class roland(BaseModel):
    def __init__(self, dataset, pretrain_model=None, meta_model=None):
        super().__init__(dataset)
        self.adj = self._make_binorm_adj(dataset.graph)
        self.edges = self.adj._indices().t()
        self.edge_norm = self.adj._values()

        # this is maintained and updated for each t, initialized by pretrained model
        self.gru = GRUCell(self.emb_size, self.emb_size)

        self.meta_model = meta_model

        # t1, initialize meta model with pretrained model
        if pretrain_model is not None:
            user_emb, item_emb = pretrain_model.generate()
            self.user_embedding = nn.Parameter(user_emb).requires_grad_(True)
            self.item_embedding = nn.Parameter(item_emb).requires_grad_(True)
        
        elif meta_model is not None:
            self.load_state_dict(meta_model.state_dict(), strict=False)
            user_emb, item_emb = meta_model.generate_lgn()
            self.user_embedding = nn.Parameter(user_emb).requires_grad_(True)
            self.item_embedding = nn.Parameter(item_emb).requires_grad_(True)

        self.edge_dropout = EdgelistDrop()
    
    def update_meta_model(self, model, meta_sd):
        if "gru.weight_ih" not in meta_sd:
            sd = model.state_dict()
            nsd = {
                "user_embedding": sd["user_embedding"],
                "item_embedding": sd["item_embedding"],
            }
        else:
            nsd = model.state_dict()
        print("nsd", nsd.keys())
        print("meta sd", meta_sd.keys())
        new_sd = average_state_dict(nsd, meta_sd, 0.9)
        # print(self.meta_model_sd)
        # print(last_model.state_dict())
        print("loading state dict:", new_sd.keys())
        print("last model state dict:", model.state_dict().keys())
        model.load_state_dict(new_sd, strict=False)
        return model

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
    
    def forward_lgn(self, edges, edge_norm, return_layers=False):
        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        res_emb = [all_emb]
        for l in range(args.num_layers):
            all_emb = self._agg(res_emb[-1], edges, edge_norm)
            res_emb.append(all_emb)
        if not return_layers:
            res_emb = sum(res_emb)
            user_res_emb, item_res_emb = res_emb.split([self.num_users, self.num_items], dim=0)
        else:
            user_res_emb, item_res_emb = [], []
            for emb in res_emb:
                u_emb, i_emb = emb.split([self.num_users, self.num_items], dim=0)
                user_res_emb.append(u_emb)
                item_res_emb.append(i_emb)
        return user_res_emb, item_res_emb

    def forward(self, edges, edge_norm, return_layers=False):
        last_user_emb, last_item_emb = self.meta_model.generate_lgn(return_layers=True)
        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        res_emb = [all_emb]
        for l in range(args.num_layers):
            all_emb = self._agg(res_emb[-1], edges, edge_norm)
            last_emb_i = torch.cat([last_user_emb[l+1], last_item_emb[l+1]], dim=0)
            all_emb = self.gru(all_emb, last_emb_i)
            res_emb.append(all_emb)
        if not return_layers:
            res_emb = sum(res_emb)
            user_res_emb, item_res_emb = res_emb.split([self.num_users, self.num_items], dim=0)
        else:
            user_res_emb, item_res_emb = [], []
            for emb in res_emb:
                u_emb, i_emb = emb.split([self.num_users, self.num_items], dim=0)
                user_res_emb.append(u_emb)
                item_res_emb.append(i_emb)
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
    def generate(self, return_layers=False):
        return self.forward(self.edges, self.edge_norm, return_layers=return_layers)
    
    @torch.no_grad()
    def generate_lgn(self, return_layers=False):
        return self.forward_lgn(self.edges, self.edge_norm, return_layers=return_layers)
    
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
