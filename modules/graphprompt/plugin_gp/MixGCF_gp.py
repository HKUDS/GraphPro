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

init = nn.init.xavier_uniform_

def cal_infonce(view1, view2, temperature, b_cos = True):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score+10e-6)
    return torch.mean(cl_loss)


class BaseModel_1(BaseModel):
    def __init__(self, dataset, pretrained_model=None, phase='finetune'):
        super().__init__(dataset)
        self.adj = self._make_binorm_adj(dataset.graph)
        self.edges = self.adj._indices().t()
        self.edge_norm = self.adj._values()

        self.n_negs = args.n_negs

        self.phase = phase

        if self.phase == 'finetune':
            pre_user_emb, pre_item_emb = pretrained_model.generate()
            self.user_embedding = nn.Parameter(pre_user_emb).requires_grad_(True)
            self.item_embedding = nn.Parameter(pre_item_emb).requires_grad_(True)

            if args.f_model == 'graphprompt':
                self.prompt_vec = nn.Parameter(init(torch.empty(1, args.emb_size)))
                self.prompt_func = lambda x: torch.mul(x, self.prompt_vec)
            
            elif args.f_model == 'gpf':
                self.prompt_vec = nn.Parameter(init(torch.empty(1, args.emb_size)))
                self.prompt_func = lambda x: x + self.prompt_vec

        self.edge_dropout = EdgelistDrop()

    def _agg(self, all_emb, edges, edge_norm):
        src_emb = all_emb[edges[:, 0]]

        # bi-norm
        src_emb = src_emb * edge_norm.unsqueeze(1)

        # conv
        if args.f_model == 'graphprompt':
            src_emb = self.prompt_func(src_emb)
        dst_emb = scatter_sum(src_emb, edges[:, 1], dim=0, dim_size=self.num_users+self.num_items)
        return dst_emb
    
    def _edge_binorm(self, edges):
        user_degs = scatter_add(torch.ones_like(edges[:, 0]), edges[:, 0], dim=0, dim_size=self.num_users)
        user_degs = user_degs[edges[:, 0]]
        item_degs = scatter_add(torch.ones_like(edges[:, 1]), edges[:, 1], dim=0, dim_size=self.num_items)
        item_degs = item_degs[edges[:, 1]]
        norm = torch.pow(user_degs, -0.5) * torch.pow(item_degs, -0.5)
        return norm
    
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


class MixGCF(BaseModel_1):
    def __init__(self, dataset, pretrained_model=None):
        super().__init__(dataset, pretrained_model)

    def forward(self, edges, edge_norm, return_res_emb=False):
        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        if args.f_model == 'gpf':
            all_emb = self.prompt_func(all_emb)
        res_emb = [all_emb]
        for l in range(args.num_layers):
            all_emb = self._agg(all_emb, edges, edge_norm)
            res_emb.append(all_emb)
        user_res_emb, item_res_emb = sum(res_emb).split([self.num_users, self.num_items], dim=0)
        if return_res_emb:
            return user_res_emb, item_res_emb, res_emb
        return user_res_emb, item_res_emb

    def negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        print(f"p_e: {p_e.shape}")

        """positive mixing"""
        seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)  # (0, 1)
        print(f"seed: {seed.shape}")
        n_e = item_gcn_emb[neg_candidates].view(batch_size, args.n_negs, -1, args.emb_size)  # [batch_size, n_negs, n_hops, channel]
        print(f"n_e: {n_e.shape}")
        n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e  # mixing

        """hop mixing"""
        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        return neg_items_emb_[[[i] for i in range(batch_size)],
                              range(neg_items_emb_.shape[1]), indices, :]

    def cal_loss(self, batch_data):
        edges, dropout_mask = self.edge_dropout(self.edges, 1-args.edge_dropout, return_mask=True)
        edge_norm = self.edge_norm[dropout_mask]

        # forward
        # neg_items: B, n_negs
        users, pos_items, neg_items = batch_data
        # print(f"neg_items: {neg_items.shape}")
        user_emb, item_emb, res_emb = self.forward(edges, edge_norm, return_res_emb=True)
        user_stack_emb, item_stack_emb = torch.stack(res_emb, dim=1).split([self.num_users, self.num_items], dim=0)
        neg_item_emb = self.negative_sampling(user_stack_emb, item_stack_emb, users, neg_items, pos_items).sum(dim=1)
        batch_user_emb = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        rec_loss = self._bpr_loss(batch_user_emb, pos_item_emb, neg_item_emb)
        reg_loss = args.weight_decay * self._reg_loss(users, pos_items, neg_items)

        loss = rec_loss + reg_loss
        loss_dict = {
            "rec_loss": rec_loss.item(),
            "reg_loss": reg_loss.item(),
        }
        return loss, loss_dict