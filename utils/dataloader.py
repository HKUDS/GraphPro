from utils.parse_args import args
from os import path
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import torch
import logging
import networkx as nx
from copy import deepcopy
from collections import defaultdict
import pandas as pd
# from torch_geometric.data import Data
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger('train_logger')
logger.setLevel(logging.INFO)

class EdgeListData:
    def __init__(self, train_file, test_file, phase='pretrain', pre_dataset=None, user_hist_files=[], has_time=True):
        logger.info(f"Loading dataset for {phase}...")
        self.phase = phase
        self.has_time = has_time
        self.pre_dataset = pre_dataset

        self.hour_interval = args.hour_interval_pre if phase == 'pretrain' else args.hour_interval_f

        self.edgelist = []
        self.edge_time = []
        self.num_users = 0
        self.num_items = 0
        self.num_edges = 0

        self.train_user_dict = {}
        self.test_user_dict = {}

        self._load_data(train_file, test_file, has_time)

        if phase == 'pretrain':
            self.user_hist_dict = self.train_user_dict
        elif phase == 'finetune':
            self.user_hist_dict = deepcopy(self.train_user_dict)
            self._load_user_hist_from_files(user_hist_files)
        
        users_has_hist = set(list(self.user_hist_dict.keys()))
        all_users = set(list(range(self.num_users)))
        users_no_hist = all_users - users_has_hist
        logger.info(f"Number of users from all users with no history: {len(users_no_hist)}")
        for u in users_no_hist:
            self.user_hist_dict[u] = []

    def _read_file(self, train_file, test_file, has_time=True):
        with open(train_file, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                if not has_time:
                    user, items = line[:2]
                    times = " ".join(["0"] * len(items.split(" ")))
                else:
                    user, items, times = line
                    
                for i in items.split(" "):
                    self.edgelist.append((int(user), int(i)))
                for i in times.split(" "):
                    self.edge_time.append(int(i))
                self.train_user_dict[int(user)] = [int(i) for i in items.split(" ")]

        self.test_edge_num = 0
        with open(test_file, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                user, items = line[:2]
                self.test_user_dict[int(user)] = [int(i) for i in items.split(" ")]
                self.test_edge_num += len(self.test_user_dict[int(user)])
        logger.info('Number of test users: {}'.format(len(self.test_user_dict)))
    
    def _read_pd(self, train_pd, test_pd, has_time=True):
        for i in range(len(train_pd)):
            line = train_pd.iloc[i]
            if not has_time:
                user, items = line[0], line[1]
                times = " ".join(["0"] * len(items.split(" ")))
            else:
                user, items, times = line[0], line[1], line[2]
                
            for i in items.split(" "):
                self.edgelist.append((int(user), int(i)))
            for i in times.split(" "):
                self.edge_time.append(int(i))
            self.train_user_dict[int(user)] = [int(i) for i in items.split(" ")]

        self.test_edge_num = 0
        for i in range(len(test_pd)):
            line = test_pd.iloc[i]
            user, items = line[0], line[1]
            self.test_user_dict[int(user)] = [int(i) for i in items.split(" ")]
            self.test_edge_num += len(self.test_user_dict[int(user)])
        logger.info('Number of test users: {}'.format(len(self.test_user_dict)))

    def _load_data(self, train_file, test_file, has_time=True):
        if isinstance(train_file, pd.DataFrame):
            self._read_pd(train_file, test_file, has_time)
        else:
            self._read_file(train_file, test_file, has_time)

        self.edgelist = np.array(self.edgelist, dtype=np.int32)
        # refine timestamp to predefined time steps at intervals
        # 0 as padding for self-loop
        self.edge_time = 1 + self.timestamp_to_time_step(np.array(self.edge_time, dtype=np.int32))
        self.num_edges = len(self.edgelist)
        if self.pre_dataset is not None:
            self.num_users = self.pre_dataset.num_users
            self.num_items = self.pre_dataset.num_items
        else:
            self.num_users = max([np.max(self.edgelist[:, 0]) + 1, np.max(list(self.test_user_dict.keys())) + 1])
            self.num_items = max([np.max(self.edgelist[:, 1]) + 1, np.max([np.max(self.test_user_dict[u]) for u in self.test_user_dict.keys()]) + 1])

        logger.info('Number of users: {}'.format(self.num_users))
        logger.info('Number of items: {}'.format(self.num_items))
        logger.info('Number of edges: {}'.format(self.num_edges))

        self.graph = sp.coo_matrix((np.ones(self.num_edges), (self.edgelist[:, 0], self.edgelist[:, 1])), shape=(self.num_users, self.num_items))

        if self.has_time:
            self.edge_time_dict = defaultdict(dict)
            for i in range(len(self.edgelist)):
                self.edge_time_dict[self.edgelist[i][0]][self.edgelist[i][1]+self.num_users] = self.edge_time[i]
                self.edge_time_dict[self.edgelist[i][1]+self.num_users][self.edgelist[i][0]] = self.edge_time[i]
                # self.edge_time_dict[self.edgelist[i][0]][self.edgelist[i][0]] = 0
                # self.edge_time_dict[self.edgelist[i][1]][self.edgelist[i][1]] = 0


        # homogenous edges generation
        # if args.ab in ["homo", "full"]:
        #     self.ii_adj = self.graph.T.dot(self.graph).tocoo()
        #     # sort the values in the sparse matrix and get top x% values and corresponding edges
        #     percentage_ii = 0.01
        #     tmp_data = sorted(self.ii_adj.data, reverse=True)
        #     tmp_data_xpercent, tmp_data_xpercent_len = tmp_data[int(len(tmp_data) * percentage_ii)], int(len(tmp_data) * percentage_ii)
        #     self.ii_adj.data = np.where(self.ii_adj.data > tmp_data_xpercent, self.ii_adj.data, 0)
        #     self.ii_adj.eliminate_zeros()
        #     logger.info(f"Sampled {len(self.ii_adj.data)} i-i edges from all {len(tmp_data)} edges.")

        #     self.uu_adj = self.graph.dot(self.graph.T).tocoo()
        #     # same filtering for uu_adj
        #     percentage_uu = 0.01
        #     tmp_data = sorted(self.uu_adj.data, reverse=True)
        #     tmp_data_xpercent, tmp_data_xpercent_len = tmp_data[int(len(tmp_data) * percentage_uu)], int(len(tmp_data) * percentage_uu)
        #     self.uu_adj.data = np.where(self.uu_adj.data > tmp_data_xpercent, self.uu_adj.data, 0)
        #     self.uu_adj.eliminate_zeros()
        #     logger.info(f"Sampled {len(self.uu_adj.data)} u-u edges from all {len(tmp_data)} edges.")


        # self.graph = nx.Graph()
        # for i in range(len(self.edgelist)):
        #     self.graph.add_edge(self.edgelist[i][0], self.edgelist[i][1], time=self.edge_time[i])
        # print(self.graph.number_of_nodes(), self.graph.number_of_edges())
        # self.ui_adj = nx.adjacency_matrix(self.graph, weight=None)[:self.num_users, self.num_users:].tocoo()
        # self.ii_adj = self.ui_adj.T.dot(self.ui_adj).tocoo()
        # self.uu_adj = self.ui_adj.dot(self.ui_adj.T).tocoo()

        # self.graph = Data(torch.zeros(self.num_users + self.num_items, 1), torch.tensor(self.edgelist).t(), torch.tensor(self.edge_time))
    
    def _load_user_hist_from_files(self, user_hist_files):
        for file in user_hist_files:
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip().split('\t')
                    user, items = int(line[0]), [int(i) for i in line[1].split(" ")]
                    try:
                        self.user_hist_dict[user].extend(items)
                    except KeyError:
                        self.user_hist_dict[user] = items

    def sample_subgraph(self):
        pass

    def get_train_batch(self, start, end):

        def negative_sampling(user_item, train_user_set, n=1):
            neg_items = []
            for user, _ in user_item:
                user = int(user)
                for i in range(n):
                    while True:
                        neg_item = np.random.randint(low=0, high=self.num_items, size=1)[0]
                        if neg_item not in train_user_set[user]:
                            break
                    neg_items.append(neg_item)
            return neg_items

        ui_pairs = self.edgelist[start:end]
        users = torch.LongTensor(ui_pairs[:, 0]).to(args.device)
        pos_items = torch.LongTensor(ui_pairs[:, 1]).to(args.device)
        if args.model == "MixGCF":
            neg_items = negative_sampling(ui_pairs, self.train_user_dict, args.n_negs)
        else:
            neg_items = negative_sampling(ui_pairs, self.train_user_dict, 1)
        neg_items = torch.LongTensor(neg_items).to(args.device)
        return users, pos_items, neg_items

    def shuffle(self):
        random_idx = np.random.permutation(self.num_edges)
        self.edgelist = self.edgelist[random_idx]
        self.edge_time = self.edge_time[random_idx]
        
    def _generate_binorm_adj(self, edgelist):
        adj = sp.coo_matrix((np.ones(len(edgelist)), (edgelist[:, 0], edgelist[:, 1])),
                            shape=(self.num_users, self.num_items), dtype=np.float32)
        
        a = sp.csr_matrix((self.num_users, self.num_users))
        b = sp.csr_matrix((self.num_items, self.num_items))
        adj = sp.vstack([sp.hstack([a, adj]), sp.hstack([adj.transpose(), b])])
        adj = (adj != 0) * 1.0
        degree = np.array(adj.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        adj = adj.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

        ui_adj = adj.tocsr()[:self.num_users, self.num_users:].tocoo()
        return adj

    def timestamp_to_time_step(self, timestamp_arr, least_time=None):
        interval_hour = self.hour_interval
        if least_time is None:
            least_time = np.min(timestamp_arr)
            print("Least time: ", least_time)
            print("2nd least time: ", np.sort(timestamp_arr)[1])
            print("3rd least time: ", np.sort(timestamp_arr)[2])
            print("Max time: ", np.max(timestamp_arr))
        timestamp_arr = timestamp_arr - least_time
        timestamp_arr = timestamp_arr // (interval_hour * 3600)
        return timestamp_arr


if __name__ == '__main__':
    edgelist_dataset = EdgeListData("dataset/yelp_small/train.txt", "dataset/yelp_small/test.txt")