"""
Adaptation of WILDSDataset class based on
https://github.com/p-lambda/wilds/blob/b38304bb6ac3b3f9326cf028d77be9f0cb7c8cdb/wilds/datasets/civilcomments_dataset.py
"""

import argparse
from pathlib import Path
from tqdm import tqdm
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
from datasets import CreoleJsonDataset, CreoleDataset, SinglishSMSDataset, CreoleDatasetWILDS
from utils import save

from collections import Counter

import random
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--file_path", type=str, default="",
                        help="Path to the data you are trying to finetune on or evaluate from")
    parser.add_argument("--split_scheme", type=str, default="standard",
                        help="Choices are dataset specific")
    parser.add_argument("--creole", type=str, default="", choices=["singlish", "haitian", "naija"])
    parser.add_argument("--creole_only", type=bool, default=False, choices=[True, False])
    # Model
    parser.add_argument("--tokenizer", type=str, default='bert-base-uncased',
                        help="Pretrained BERT: bert-base-uncased, bert-base-multilingual-cased, xlm-roberta-base, etc.")
    parser.add_argument("--from_pretrained", type=str, default='bert-base-uncased',
                        help="Pretrained BERT: bert-base-uncased, bert-base-multilingual-cased, xlm-roberta-base, etc.,"
                             "Or full path to our pretrained model.")
    parser.add_argument("--base_lang", type=str, default="en",
                        help="Base language of the Creole")
    parser.add_argument("--group_strategy", type=str, default="collect", choices=["collect", "cluster", "percent", "random", "one", "language"])
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
    parser.add_argument("--num_epochs", type=int, default=100)
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
        'loss_function', 'val_metric', 'val_metric_decreasing', 'num_epochs', 'optimizer', 'learning_rate', 'weight_decay',
    ]

    config.optimizer_kwargs = {'eps': 1e-8}
    config.scheduler_kwargs = {'num_warmup_steps': 0}
   
    for field in required_fields:
        assert getattr(config, field) is not None, f"Must manually specify {field} for this setup."

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, do_lower_case=('uncased' in config.from_pretrained))
    vocab_size = tokenizer.vocab_size

    base_dataset = CreoleJsonDataset(src_file=config.file_path, tokenizer=tokenizer, base_language=config.base_lang, creole_only=config.creole_only)

    """ 
    if config.creole == "singlish":
        base_dataset = SinglishSMSDataset(src_file=config.file_path, tokenizer=tokenizer)
    else:
        base_dataset = CreoleDataset(src_file=config.file_path, tokenizer=tokenizer, base_language=config.base_lang)
    """
    train_dataset = CreoleDatasetWILDS(base_dataset, tokenizer, group_strategy=config.group_strategy, group_file=config.file_path, creole=config.creole) #this one has (x, y, metadata)
    #FIXME: DRO+language will not work with creole-only option presently!!! See dataset.py
    train_grouper = CombinatorialGrouper(dataset=train_dataset, groupby_fields=train_dataset.metadata_fields)
    group_ids = train_grouper.metadata_to_group(train_dataset.metadata_array)
    
    #base_dev_dataset =  CreoleDataset(src_file=config.file_path, tokenizer=tokenizer, nexamples=15)
    #dev_dataset = CreoleDatasetWILDS(base_dev_dataset, tokenizer)
    #dev_grouper = CombinatorialGrouper(dataset=dev_dataset, groupby_fields=['eng', 'chn'])
    #dev_group_ids = dev_grouper.metadata_to_group(dev_dataset.metadata_array)

    batch_sampler = GroupSampler(
        group_ids=group_ids,
        batch_size=config.batch_size,
        n_groups_per_batch=train_grouper.n_groups,
        uniform_over_groups=False, #was True
        distinct_groups=False) #was True
    torch.set_printoptions(threshold=100)
    print(f"group_ids: {group_ids} | num groups: {train_grouper.n_groups}")
    print(f"size of groups: {group_ids.size()}")
    print(Counter(group_ids.tolist()))
    train_loader = DataLoader(train_dataset, shuffle=False, sampler=None, batch_sampler=batch_sampler, drop_last=False)

    #dev_loader = DataLoader(dev_dataset, shuffle=False, sampler=None, batch_size=8)    

    print(f"metadata_array: {train_dataset.metadata_array}")
    train_g = train_grouper.metadata_to_group(train_dataset.metadata_array)
    is_group_in_train = get_counts(train_g, train_grouper.n_groups) > 0
    print(f"is_group_in_train: {is_group_in_train}")
    #init DRO algorithm
    model = AutoModelForMaskedLM.from_pretrained(config.from_pretrained).to(config.device)

    #options for losses and metric
    losses = {
        'cross_entropy': ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100)),
        'mse': MSE(name='loss'),
        'multitask_bce': MultiTaskLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none')),
    }

    algo_log_metrics = {
        'accuracy': Accuracy(),
        'mse': MSE(),
        'multitask_accuracy': MultiTaskAccuracy(),
        #'f1': F1(average='macro'),
        None: None,
    }

    algorithm = GroupDRO(
        config=config,
        model=model,
        d_out=train_dataset.y_size,
        grouper=train_grouper,
        loss=losses[config.loss_function], #cross_entropy
        metric=None, #MSE(),  #
        n_train_steps= 100000, #len(train_loader)*config.num_epochs,
        is_group_in_train=is_group_in_train)

    #model.eval()
    #model.cpu()
    #fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    #print(fill_mask("Pop [MASK] 'N' Roll.")) 
    #del fill_mask

    model.cuda()
    algorithm.train()
    
    #dev_iterator = tqdm(dev_loader)

    epoch_y_true = []
    epoch_y_pred = []
    epoch_metadata = []

    dirLUT = {'bert-base-uncased': 'bert', 'bert-base-multilingual-cased': 'mbert', 'xlm-roberta-base': 'xlmr',
              'prajjwal1/bert-tiny': 'tinybert', 'prajjwal1/bert-small': 'smallbert'}
   
    steps_so_far = 0 
    for i in range(0, config.num_epochs):
        for batch in tqdm(train_loader):
            batch_results = algorithm.update(batch, vocab_size) #FIXME ADD VOCAB
            steps_so_far += 1
            # These tensors are already detached, but we need to clone them again
            # Otherwise they don't get garbage collected properly in some versions
            # The subsequent detach is just for safety
            # (they should already be detached in batch_results)
            # epoch_y_true.append(batch_results['y_true'].clone().detach())
            # epoch_y_pred.append(batch_results['y_pred'][0].clone().detach())
            # epoch_metadata.append(batch_results['metadata'].clone().detach())
            if steps_so_far % 20000 == 0 and steps_so_far !=0:
                # Save the model
                print(f"Saving the model at step {steps_so_far}... ")
                path_to_checkpoint = f"{config.checkpoint_dir}/{dirLUT[config.tokenizer]}/{config.creole}"
                Path(path_to_checkpoint).mkdir(parents=True, exist_ok=True)
                file_name = f'dro_{config.creole}_{config.group_strategy}_{steps_so_far}.pth'
                full_path_to_file = os.path.join(path_to_checkpoint, file_name)
                save(algorithm, steps_so_far, full_path_to_file)
            if steps_so_far == 100000:
                path_to_checkpoint = f"{config.checkpoint_dir}/{dirLUT[config.tokenizer]}/{config.creole}"
                Path(path_to_checkpoint).mkdir(parents=True, exist_ok=True)
                file_name = f'dro_{config.creole}_{config.group_strategy}_{steps_so_far}.pth'
                full_path_to_file = os.path.join(path_to_checkpoint, file_name)
                save(algorithm,steps_so_far, full_path_to_file)
                print(f"TRAINING REACHED 100k steps! Saved and exiting")
                exit(100000)

    #i #algorithm.eval()
    # #for batch in dev_iterator:
    # model.eva
    print("done training! ")
    #trainer.save_model(args.checkpoint_dir)
    # save(algorithm, i, os.path.join(config.checkpoint_dir, f'dro_{config.creole}_{config.group_strategy}_last_model.pth'))
    # model.cpu()
    # fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    # print(fill_mask("Pop [MASK] 'N' Roll."))
    # del fill_mask
    # model.cuda()
    # model.train()
    # #algorithm.train()
        


if __name__ == "__main__":
    main()
