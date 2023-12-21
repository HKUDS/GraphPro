import torch
import torch.nn as nn
from utils.parse_args import args
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, dataloader):
        super(BaseModel, self).__init__()
        self.num_users = dataloader.num_users
        self.num_items = dataloader.num_items
        self.emb_size = args.emb_size

    def forward(self):
        pass

    def cal_loss(self, batch_data):
        pass

    def _check_inf(self, loss, pos_score, neg_score, edge_weight):
        # find inf idx
        inf_idx = torch.isinf(loss) | torch.isnan(loss)
        if inf_idx.any():
            print("find inf in loss")
            if type(edge_weight) != int:
                print(edge_weight[inf_idx])
            print(f"pos_score: {pos_score[inf_idx]}")
            print(f"neg_score: {neg_score[inf_idx]}")
            raise ValueError("find inf in loss")

    def _make_binorm_adj(self, mat):
        a = csr_matrix((self.num_users, self.num_users))
        b = csr_matrix((self.num_items, self.num_items))
        mat = sp.vstack(
            [sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        # mat = (mat + sp.eye(mat.shape[0])) * 1.0# MARK
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        mat = mat.dot(d_inv_sqrt_mat).transpose().dot(
            d_inv_sqrt_mat).tocoo()

        # make torch tensor
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).to(args.device)
    
    def _make_binorm_adj_self_loop(self, mat):
        a = csr_matrix((self.num_users, self.num_users))
        b = csr_matrix((self.num_items, self.num_items))
        mat = sp.vstack(
            [sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0 # self loop
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        mat = mat.dot(d_inv_sqrt_mat).transpose().dot(
            d_inv_sqrt_mat).tocoo()

        # make torch tensor
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).to(args.device)


    def _sp_matrix_to_sp_tensor(self, sp_matrix):
        coo = sp_matrix.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(indices, values, coo.shape).coalesce().to(args.device)

    def _bpr_loss(self, user_emb, pos_item_emb, neg_item_emb):
        pos_score = (user_emb * pos_item_emb).sum(dim=1)
        neg_score = (user_emb * neg_item_emb).sum(dim=1)
        loss = -torch.log(1e-10 + torch.sigmoid((pos_score - neg_score)))
        self._check_inf(loss, pos_score, neg_score, 0)
        return loss.mean()
    
    def _nce_loss(self, pos_score, neg_score, edge_weight=1):
        numerator = torch.exp(pos_score)
        denominator = torch.exp(pos_score) + torch.exp(neg_score).sum(dim=1)
        loss = -torch.log(numerator/denominator) * edge_weight
        self._check_inf(loss, pos_score, neg_score, edge_weight)
        return loss.mean()
    
    def _infonce_loss(self, pos_1, pos_2, negs, tau):
        pos_1 = self.cl_mlp(pos_1)
        pos_2 = self.cl_mlp(pos_2)
        negs = self.cl_mlp(negs)
        pos_1 = F.normalize(pos_1, dim=-1)
        pos_2 = F.normalize(pos_2, dim=-1)
        negs = F.normalize(negs, dim=-1)
        pos_score = torch.mul(pos_1, pos_2).sum(dim=1)
        # B, 1, E * B, E, N -> B, N
        neg_score = torch.bmm(pos_1.unsqueeze(1), negs.transpose(1, 2)).squeeze(1)
        # infonce loss
        numerator = torch.exp(pos_score / tau)
        denominator = torch.exp(pos_score / tau) + torch.exp(neg_score / tau).sum(dim=1)
        loss = -torch.log(numerator/denominator)
        self._check_inf(loss, pos_score, neg_score, 0)
        return loss.mean()
    