import torch
import torch.nn as nn
from modules.base_model import BaseModel
from utils.parse_args import args
import torch.nn.functional as F
from modules.utils import EdgelistDrop
import logging
from modules.utils import scatter_add, scatter_sum
from torch_scatter import scatter_softmax

init = nn.init.xavier_uniform_
logger = logging.getLogger('train_logger')

class GraphPro(BaseModel):
    def __init__(self, dataset, pretrained_model=None, phase='pretrain'):
        super().__init__(dataset)
        self.adj = self._make_binorm_adj(dataset.graph)
        self.edges = self.adj._indices().t()
        self.edge_norm = self.adj._values()

        self.edge_times = [dataset.edge_time_dict[e[0]][e[1]] for e in self.edges.cpu().tolist()]
        self.edge_times = torch.LongTensor(self.edge_times).to(args.device)

        self.phase = phase

        if self.phase == 'pretrain' or self.phase == 'for_tune':
            self.user_embedding = nn.Parameter(init(torch.empty(self.num_users, self.emb_size)))
            self.item_embedding = nn.Parameter(init(torch.empty(self.num_items, self.emb_size)))

            if self.phase == 'for_tune':
                self.emb_gate = self.random_gate
            elif self.phase == 'pretrain':
                self.emb_gate = lambda x: x # no gating

        # load gating weights from pretrained model
        elif self.phase == 'finetune':
            pre_user_emb, pre_item_emb = pretrained_model.generate()

            self.user_embedding = nn.Parameter(pre_user_emb).requires_grad_(True)
            self.item_embedding = nn.Parameter(pre_item_emb).requires_grad_(True)

            self.gating_weight = nn.Parameter(init(torch.empty(args.emb_size, args.emb_size)))
            self.gating_bias = nn.Parameter(init(torch.empty(1, args.emb_size)))

            self.emb_dropout = nn.Dropout(args.emb_dropout)

            self.emb_gate = lambda x: self.emb_dropout(torch.mul(x, torch.sigmoid(torch.matmul(x, self.gating_weight) + self.gating_bias)))
        
        self.edge_dropout = EdgelistDrop()

        logger.info(f"Max Time Step: {self.edge_times.max()}")
    
    def random_gate(self, x):
        gating_weight = F.normalize(torch.randn((args.emb_size, args.emb_size)).to(args.device))
        gating_bias = F.normalize(torch.randn((1, args.emb_size)).to(args.device))

        gate = torch.sigmoid(torch.matmul(x, gating_weight) + gating_bias)

        return torch.mul(x, gate)

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
    
    def _relative_edge_time_encoding(self, edges, edge_times, max_step=None):
        # for each node, normalize edge_times according to its neighbors
        # edge_times: [E]
        # rescal to 0-1
        edge_times = edge_times.float()
        if max_step is None:
            max_step = edge_times.max()
        edge_times = (edge_times - edge_times.min()) / (max_step - edge_times.min())
        # edge_times = torch.exp(edge_times)
        # edge_times = torch.sigmoid(edge_times)
        dst_nodes = edges[:, 1]
        time_norm = scatter_softmax(edge_times, dst_nodes, dim_size=self.num_users+self.num_items)
        # time_norm = time_norm[dst_nodes]
        return time_norm

    def forward(self, edges, edge_norm, edge_times, max_time_step=None):
        time_norm = self._relative_edge_time_encoding(edges, edge_times, max_step=max_time_step)
        edge_norm = edge_norm * 1/2 + time_norm * 1/2

        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        all_emb = self.emb_gate(all_emb)
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
        edge_times = self.edge_times[dropout_mask]

        # forward
        users, pos_items, neg_items = batch_data
        user_emb, item_emb = self.forward(edges, edge_norm, edge_times)
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
    def generate(self, max_time_step=None):
        return self.forward(self.edges, self.edge_norm, self.edge_times, max_time_step=max_time_step)
    
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
