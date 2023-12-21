import os
from os import path
import logging
import datetime
import sys

def get_local_time():
    return datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')

def log_exceptions(func):
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('train_logger')
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            raise e
    return wrapper

class Logger(object):
    def __init__(self, args, log_configs=True):
        self.logger = logging.getLogger('train_logger')
        self.logger.setLevel(logging.INFO)
        self.args = args

        data_name = args.data_path.split("/")[-1]

        save_dir = path.join(args.save_path, data_name, args.exp_name)
        if not path.exists(save_dir):
            os.makedirs(save_dir)
        args.save_dir = save_dir

        if args.log:
            cur_time = get_local_time()
            args.exp_time = cur_time
            log_file = logging.FileHandler(path.join(save_dir, 'train_log_{}.txt'.format(cur_time)))
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            log_file.setFormatter(formatter)
            self.logger.addHandler(log_file)
            
        self.log(f"DESC: {args.desc}")
        self.log(f"PID: {os.getpid()}")

        # log command that runs the code
        s = ""
        for arg in sys.argv:
            s = s + arg + " "
        self.log(os.path.basename(sys.executable) + " " + s)

        if log_configs:
            self.log(args)
    
    def info(self, message):
        self.logger.info(message)

    def log(self, message, save_to_log=True, print_to_console=True):
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)

    def log_loss(self, epoch_idx, loss_log_dict, save_to_log=True, print_to_console=True):
        epoch = self.args.num_epochs
        message = '[Epoch {:3d} / {:3d} Training Time: {:.2f}s ] '.format(epoch_idx, epoch, loss_log_dict['train_time'])
        for loss_name in loss_log_dict:
            if loss_name == 'train_time':
                continue
            message += '{}: {:.4f} '.format(loss_name, loss_log_dict[loss_name])
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)

    def log_eval(self, eval_result, k, save_to_log=True, print_to_console=True):
        message = 'Eval Time: {:.2f}s '.format(eval_result['eval_time'])
        for metric in eval_result:
            if metric == 'eval_time':
                continue
            message += '['
            for i in range(len(k)):
                message += '{}@{}: {:.4f} '.format(metric, k[i], eval_result[metric][i])
            message += '] '
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)