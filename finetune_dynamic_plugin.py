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

modules_class = 'modules.dynamicGNN.plugin_dynamic.'+args.pre_model+"_dynamic"

def import_pretrained_model():
    module = importlib.import_module('modules.plugins.'+args.pre_model)
    return getattr(module, args.pre_model)

def import_finetune_model():
    module = importlib.import_module(modules_class)
    return getattr(module, args.pre_model+"_"+args.f_model)

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
logger.log(f"test_data_num: {test_data_num}")
test_datas = [path.join(args.data_path, f"test_{i}.txt") for i in range(1, test_data_num+1)]
all_data = [pretrain_data, pretrain_val_data, finetune_data, *test_datas]

recalls, ndcgs = [], []

if args.f_model == 'roland':
    @log_exceptions
    def run():
        pretrain_dataset = EdgeListData(pretrain_data, pretrain_val_data)

        # LightGCN
        pretrain_model = import_pretrained_model()(pretrain_dataset, phase='vanilla').to(args.device)
        pretrain_model.load_state_dict(torch.load(args.pre_model_path), strict=False)
        pretrain_model.eval()

        meta_model_sd = pretrain_model.state_dict()

        for num_stage in range(1,len(test_datas)+1):
            test_data_idx = num_stage + 2
            ft_data_idx = test_data_idx - 1
            finetune_dataset = EdgeListData(train_file=all_data[ft_data_idx], test_file=all_data[test_data_idx], phase='finetune', pre_dataset=pretrain_dataset, has_time=True, user_hist_files=all_data[:ft_data_idx])

            if num_stage == 1:
                model = import_finetune_model()(finetune_dataset, pretrain_model, meta_model=pretrain_model).to(args.device)
            if num_stage > 1:
                # update model for next stage
                model = import_finetune_model()(finetune_dataset, meta_model=updated_model).to(args.device)
                print(model)

            logger.info(f"ROLAND Learning Stage {num_stage}, test data: {all_data[test_data_idx]}, incremental train data: {all_data[ft_data_idx]}")

            trainer = Trainer(finetune_dataset, logger)
            best_perform = trainer.train_finetune(model)
            recalls.append(best_perform['recall'][0])
            ndcgs.append(best_perform['ndcg'][0])

            # update meta model
            # reload the best model
            model.load_state_dict(torch.load(trainer.save_path))
            model.meta_model = None
            updated_model = model.update_meta_model(model, meta_model_sd)
            meta_model_sd = updated_model.state_dict()
            # update exp time for saving new model
            args.exp_time = datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')

        logger.info(f"recalls: {recalls} \n ndcgs: {ndcgs} \n avg. recall: {np.round(np.mean(recalls), 4)}, avg. ndcg: {np.round(np.mean(ndcgs), 4)}")
    run()

elif args.f_model in ['evolveGCN_H', 'evolveGCN_O']:
    @log_exceptions
    def run():
        pretrain_dataset = EdgeListData(pretrain_data, pretrain_val_data)

        # LightGCN
        pretrain_model = import_pretrained_model()(pretrain_dataset, phase='vanilla').to(args.device)
        pretrain_model.load_state_dict(torch.load(args.pre_model_path), strict=False)
        pretrain_model.eval()

        for num_stage in range(1,len(test_datas)+1):
            test_data_idx = num_stage + 2
            ft_data_idx = test_data_idx - 1
            finetune_dataset = EdgeListData(train_file=all_data[ft_data_idx], test_file=all_data[test_data_idx], phase='finetune', pre_dataset=pretrain_dataset, has_time=True, user_hist_files=all_data[:ft_data_idx])

            if num_stage == 1:
                last_emb = torch.concat(pretrain_model.generate(), dim=0)
                model = import_finetune_model()(finetune_dataset, pretrain_model, last_emb).to(args.device)
            if num_stage > 1:
                # update model for next stage
                model = import_finetune_model()(finetune_dataset, last_emb=last_emb).to(args.device)
                print(model)

            logger.info(f"EvolveGCN Learning Stage {num_stage}, test data: {all_data[test_data_idx]}, incremental train data: {all_data[ft_data_idx]}")

            trainer = Trainer(finetune_dataset, logger)
            best_perform = trainer.train_finetune(model)
            recalls.append(best_perform['recall'][0])
            ndcgs.append(best_perform['ndcg'][0])
            
            if args.f_model == 'evolveGCN_H':
                last_emb = torch.concat(model.generate(), dim=0)
            elif args.f_model == 'evolveGCN_O':
                last_emb = torch.concat([model.user_embedding, model.item_embedding], dim=0)

            # update exp time for saving new model
            args.exp_time = datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')
            
        logger.info(f"recalls: {recalls} \n ndcgs: {ndcgs} \n avg. recall: {np.round(np.mean(recalls), 4)}, avg. ndcg: {np.round(np.mean(ndcgs), 4)}")

    run()