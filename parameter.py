import argparse
import logging
import os
import sys

def setuplogging(args, rank=0):
    root = logging.getLogger()
    if len(root.handlers)<=1:
        root.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(f"[{rank}] [%(levelname)s %(asctime)s] %(message)s")
        handler.setFormatter(formatter)
        root.addHandler(handler)

        fh = logging.FileHandler(os.path.join(args.log_dir,'logging_file.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        root.addHandler(fh)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "ag_news":('text',None),
    "smm4h":('text',None),
    "mnist":('image'),
    "cifar10":('image')
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument('--mode',type=str,default='train')
    parser.add_argument('--log_dir',type=str,default=None)
    parser.add_argument('--model_dir',type=str,default=None)
    parser.add_argument('--ema_weight',type=float,default=0.95)
    parser.add_argument("--debug",type=str2bool,default=False)
    parser.add_argument("--debug_step",type=int,default=1)
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument('--local_pretrain',type=int,default=20)
    parser.add_argument('--init_model',type=str,default=None)
    parser.add_argument('--threshold',type=float,default=0.8)
    parser.add_argument('--entropy_threshold',type=float,default=0.1)
    parser.add_argument('--temperature',type=float,default=1.7)
   
    parser.add_argument(
        "--learning_rate_sup",
        type=float,
        default=2e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_unsup",
        type=float,
        default=2e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--label_ratio",type=float,default=0.05)
    
    parser.add_argument("--num_train_epochs_server", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--num_train_epochs_client", type=int, default=1, help="Total number of training epochs to perform.")
    
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.", default=None)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument('--unlabeled_data_ratio', type=float, default=1)
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="The number of layers for each portion of users")
    parser.add_argument("--tune_emb", action="store_true", default=False)
    parser.add_argument("--tune_one", action="store_true", default=False)
    parser.add_argument("--tune_cls", action="store_true", default=False)
    parser.add_argument("--tune_pooler", action="store_true", default=False)
    parser.add_argument("--thresh_epoch", type=int, default=100)
    parser.add_argument('--alpha',type=float,default=0.5)
    parser.add_argument("--num_users", type=int, default=100, help='number of users')
    parser.add_argument("--sample_ratio", type=float, default=0.2, help='users sample ratio per round')
    parser.add_argument("--start_train_epochs",type=int,default=5)
    parser.add_argument("--enable_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_tags", nargs='+', default=[], help='wandb tags')
    parser.add_argument("--wandb_name", type=str, default='', help='wandb name')
    parser.add_argument("--wandb_group", type=str, default='ab_remove', help='wandb group')
    parser.add_argument("--wandb_suffix", type=str, default='', help='wandb suffix')
    parser.add_argument('--wandb_project', type=str, default='', help='wandb suffix')
    parser.add_argument('--sup_epoch', type=int, default=1)
    parser.add_argument('--semi_epoch', type=int, default=1)
    parser.add_argument("--rounds", type=int, default=500, help='rounds of FL')
    parser.add_argument("--save_per_round", type=bool, default=False, help='save per round')
    parser.add_argument("--log_round", type=int, default=1, help='log round')
    parser.add_argument("--log_file",type=str,default='./log.txt')
    # parser.add_argument("--isolate_hete", action="store_true", default=False, help='default false to enable hete')
    parser.add_argument("--cache_dir", type=str, default="../cache", help="cache data dir.")

    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args