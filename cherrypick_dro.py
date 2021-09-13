import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM, AdamW, AutoConfig
from transformers import pipeline

from wilds.common.grouper import CombinatorialGrouper
from wilds.common.data_loaders import GroupSampler
from wilds.common.utils import get_counts
from wilds.common.metrics.loss import ElementwiseLoss, Loss, MultiTaskLoss
from wilds.common.metrics.all_metrics import Accuracy, MultiTaskAccuracy, MSE

from algorithms.groupDRO import GroupDRO
from datasets import CreoleDataset, CreoleDatasetWILDS
from utils import load

from collections import Counter

import random
import numpy as np

random.seed(28)
np.random.seed(28)
torch.manual_seed(28)


"""
We just need to load the dataloaders and stuff
And path to checkpoint
"""

def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--file_path", type=str, default="",
                        help="Path to the data you are trying to finetune on or evaluate from")
    parser.add_argument("--split_scheme", type=str, default="standard",
                        help="Choices are dataset specific")
    parser.add_argument("--creole", type=str, default="", choices=["singlish", "haitian", "naija"])

    # Model
    parser.add_argument("--tokenizer", type=str, default='bert-base-uncased',
                        help="Pretrained BERT: bert-base-uncased, bert-base-multilingual-cased, xlm-roberta-base, etc.")
    parser.add_argument("--from_pretrained", type=str, default='bert-base-uncased',
                        help="Pretrained BERT: bert-base-uncased, bert-base-multilingual-cased, xlm-roberta-base, etc.,"
                             "Or full path to our pretrained model.")
    parser.add_argument("--base_lang", type=str, default="en",
                        help="Base language of the Creole")
    parser.add_argument("--group_strategy", type=str, default="collect", choices=["collect", "cluster", "percent", "random"])
    parser.add_argument("--group_file", type=str, default="",
                        help="Path to json file for collection or percent group strategies")

    # Logging
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--debug", default=False, action="store_true",
                        help="Enable debug-level logging.")
    # Group
    parser.add_argument("--algo_log_metric", type=str, default="mse")
    parser.add_argument("--train_loader", type=str, default="standard", choices=['standard', 'group'])
    parser.add_argument("--uniform_over_groups", default=True, action="store_true")
    parser.add_argument("--n_groups_per_batch", type=int, default=1)
    parser.add_argument("--no_group_logging", default=True, action="store_true")
    parser.add_argument("--group_dro_step_size", type=float, default=0.01)
    # Training
    parser.add_argument("--loss_function", type=str, default="cross_entropy",
                        choices=["cross_entropy", "mse", "multitask_bcd"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Former default was 5e-5")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=int, default=1.0)
    parser.add_argument("--scheduler", type=str, default="linear_schedule_with_warmup")
    parser.add_argument("--scheduler_metric_name", type=str, default="fuckoff")
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                        help="Default in run_glue.py")
    parser.add_argument("--device", type=str, default="cuda")
    # Evaluation
    parser.add_argument("--eval_loader", type=str, default="standard", choices=['standard', 'group'])
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--val_metric", type=str, default="f1", choices=["accuracy", "mse", "multitask_accruacy", "f1"])
    parser.add_argument("--val_metric_decreasing", default=False, action="store_true")

    return parser.parse_args()

def main():
    config = parse_args()
    required_fields = [
        'split_scheme', 'train_loader', 'uniform_over_groups', 'batch_size', 'eval_loader', 'from_pretrained',
        'loss_function', 'val_metric', 'val_metric_decreasing', 'num_epochs', 'optimizer', 'learning_rate',
        'weight_decay',
    ]

    config.optimizer_kwargs = {'eps': 1e-8}
    config.scheduler_kwargs = {'num_warmup_steps': 0}

    for field in required_fields:
        assert getattr(config, field) is not None, f"Must manually specify {field} for this setup."

    torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, do_lower_case=('uncased' in config.from_pretrained))
    base_dataset = CreoleDataset(src_file=config.file_path, tokenizer=tokenizer)
    train_dataset = CreoleDatasetWILDS(base_dataset, tokenizer, group_strategy=config.group_strategy,
                                       group_file=config.group_file,
                                       creole=config.creole)  # this one has (x, y, metadata)
    train_grouper = CombinatorialGrouper(dataset=train_dataset, groupby_fields=train_dataset.metadata_fields)
    group_ids = train_grouper.metadata_to_group(train_dataset.metadata_array)

    # base_dev_dataset =  CreoleDataset(src_file=config.file_path, tokenizer=tokenizer, nexamples=15)
    # dev_dataset = CreoleDatasetWILDS(base_dev_dataset, tokenizer)
    # dev_grouper = CombinatorialGrouper(dataset=dev_dataset, groupby_fields=['eng', 'chn'])
    # dev_group_ids = dev_grouper.metadata_to_group(dev_dataset.metadata_array)

    batch_sampler = GroupSampler(
        group_ids=group_ids,
        batch_size=config.batch_size,
        n_groups_per_batch=train_grouper.n_groups,
        uniform_over_groups=False,  # was True
        distinct_groups=False)  # was True
    torch.set_printoptions(threshold=100)
    print(f"group_ids: {group_ids} | num groups: {train_grouper.n_groups}")
    print(f"size of groups: {group_ids.size()}")
    print(Counter(group_ids.tolist()))
    train_loader = DataLoader(train_dataset, shuffle=False, sampler=None, batch_sampler=batch_sampler, drop_last=False)

    # dev_loader = DataLoader(dev_dataset, shuffle=False, sampler=None, batch_size=8)

    print(f"metadata_array: {train_dataset.metadata_array}")
    train_g = train_grouper.metadata_to_group(train_dataset.metadata_array)
    is_group_in_train = get_counts(train_g, train_grouper.n_groups) > 0
    print(f"is_group_in_train: {is_group_in_train}")

    # init DRO algorithm
    model = AutoModelForMaskedLM.from_pretrained(config.from_pretrained).to(config.device)

    # options for losses and metric
    losses = {
        'cross_entropy': ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100)),
        'mse': MSE(name='loss'),
        'multitask_bce': MultiTaskLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none')),
    }

    algo_log_metrics = {
        'accuracy': Accuracy(),
        'mse': MSE(),
        'multitask_accuracy': MultiTaskAccuracy(),
        # 'f1': F1(average='macro'),
        None: None,
    }

    algorithm = GroupDRO(
        config=config,
        model=model,
        d_out=train_dataset.y_size,
        grouper=train_grouper,
        loss=losses[config.loss_function],  # cross_entropy
        metric=None,  # MSE(),  #
        n_train_steps=len(train_loader) * config.num_epochs,
        is_group_in_train=is_group_in_train)

    print("--------------- BEFORE TRAINING ------------------")
    model.eval()
    model.cpu()
    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    print(fill_mask("E get one soup like [MASK] wey Mama Maro bin teach me")) #wey
    print("---")
    print(fill_mask("[MASK] soup make sense")) #Di
    print("---")
    print(fill_mask("[MASK] e do again?")) #Wetin
    print("---")
    print(fill_mask("Wetin we go [MASK]?")) #chop
    print("---")
    print(fill_mask("Nna na wetin I wan [MASK] o.")) #chop
    print("---")
    print(fill_mask("De said di guy go sue dem for dat [MASK] wey dem waka")) #waka
    del fill_mask

    path_to_checkpoint = os.path.join(config.checkpoint_dir, f'dro_{config.creole}_{config.group_strategy}_last_model.pth')

    algorithm_anew, idk = load(algorithm=algorithm, path=path_to_checkpoint)
    #print(algorithm_anew)
    #print(dir(algorithm_anew))
    #print(idk)
    #print(type(idk))
    #print(dir(idk))
    print("--------------- AFTER TRAINING ------------------")
    algorithm_anew.eval()
    algorithm_anew.cpu()
    fill_mask = pipeline("fill-mask", model=algorithm_anew.model, tokenizer=tokenizer)
    print(fill_mask("E get one soup like [MASK] wey Mama Maro bin teach me")) #wey
    print("---")
    print(fill_mask("[MASK] soup make sense")) #Di
    print("---")
    print(fill_mask("[MASK] e do again?")) #Wetin
    print("---")
    print(fill_mask("Wetin we go [MASK]?")) #chop
    print("---")
    print(fill_mask("Nna na wetin I wan [MASK] o.")) #chop
    print("---")
    print(fill_mask("De said di guy go sue dem for dat [MASK] wey dem waka")) #waka

if __name__ == "__main__":
    main()



