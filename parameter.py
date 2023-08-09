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
    "cifar10":('image'),
    "cifar100":('image'),
    "TinyImageNet":('image'),
    'svhn':('image')
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
    parser.add_argument('--log_dir',type=str,default=None)
    parser.add_argument('--model_dir',type=str,default=None)
    parser.add_argument("--debug",type=str2bool,default=False)
   
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
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--label_ratio",type=float,default=0.05)
    parser.add_argument('--unlabeled_data_ratio', type=float, default=1)
    parser.add_argument('--alpha',type=float,default=0.5)
    parser.add_argument("--num_users", type=int, default=100, help='number of users')
    parser.add_argument("--sample_ratio", type=float, default=0.2, help='users sample ratio per round')
    parser.add_argument("--rounds", type=int, default=500, help='rounds of FL')
    parser.add_argument('--ema_weight',type=float,default=0.95)
    parser.add_argument('--entropy_threshold',type=float,default=0.1)

    args = parser.parse_args()

    return args