import sys
sys.path.append('./')

from os import path
from utils.parse_args import args
from utils.dataloader import EdgeListData
from tqdm import tqdm
# import dgl
import random
import numpy as np
import torch
from utils.logger import Logger, log_exceptions
from modules.dynamicGNN.trainer_roland import Trainer
import importlib
import setproctitle
import pandas as pd
import datetime

setproctitle.setproctitle('GraphPro')

modules_class = 'modules.graphprompt.'

def import_pretrained_model():
    module = importlib.import_module('modules.LightGCN')
    return getattr(module, 'LightGCN')

def import_finetune_model():
    module = importlib.import_module(modules_class + "GP")
    return getattr(module, 'GP')

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

init_seed(args.seed)
logger = Logger(args)

pretrain_data = path.join(args.data_path, "pretrain.txt")
pretrain_val_data = path.join(args.data_path, "pretrain_val.txt")
finetune_data = path.join(args.data_path, "fine_tune.txt")
test_data_num = 8 if args.data_path.split('/')[-1] == 'amazon' else 4
test_datas = [path.join(args.data_path, f"test_{i}.txt") for i in range(1, test_data_num+1)]
all_data = [pretrain_data, pretrain_val_data, finetune_data, *test_datas]

recalls, ndcgs = [], []

if args.f_model == 'gpf' or args.f_model == 'graphprompt':
    @log_exceptions
    def run():
        pretrain_dataset = EdgeListData(pretrain_data, pretrain_val_data)

        # LightGCN
        pretrain_model = import_pretrained_model()(pretrain_dataset, phase='vanilla').to(args.device)
        pretrain_model.load_state_dict(torch.load(args.pre_model_path), strict=False)
        pretrain_model.eval()

        for num_stage in range(1, test_data_num+1):
            test_data_idx = num_stage + 2
            ft_data_idx = test_data_idx - 1
            finetune_dataset = EdgeListData(train_file=all_data[ft_data_idx], test_file=all_data[test_data_idx], phase='finetune', pre_dataset=pretrain_dataset, has_time=True, user_hist_files=all_data[:ft_data_idx])

            model = import_finetune_model()(finetune_dataset, pretrain_model).to(args.device)

            logger.info(f"Learning Stage {num_stage}, test data: {all_data[test_data_idx]}, incremental train data: {all_data[ft_data_idx]}")

            trainer = Trainer(finetune_dataset, logger)
            trainer.train_finetune(model)

            # update exp time for saving new model
            args.exp_time = datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')

    run()