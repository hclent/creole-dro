import logging
import logzero
import argparse

import os
import string
import time
import random
import numpy as np
import re
from statistics import mean
from tqdm import tqdm
from collections import Counter
import csv
from pathlib import Path


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM

from utils import load
from datasets import CreoleJsonDataset, CreoleDatasetWILDS

from wilds.common.grouper import CombinatorialGrouper
from wilds.common.data_loaders import GroupSampler
from wilds.common.utils import get_counts
from wilds.common.metrics.loss import ElementwiseLoss, MultiTaskLoss
from wilds.common.metrics.all_metrics import Accuracy, MultiTaskAccuracy, MSE

from algorithms.groupDRO import GroupDRO


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--file_path", type=str, default="",
                        help="Path to the data you are trying to finetune on or evaluate from")
    parser.add_argument("--dictionary_path", type=str, default="",
                        help="Path to the creole specific dictionary")
    parser.add_argument("--creole", type=str, default="", choices=["singlish", "haitian", "naija"])
    parser.add_argument("--experiment", type=str, default="baseline", choices=["pretrained", "baseline", "dro"])
    parser.add_argument("--group_strategy", type=str, default="collect",
                        choices=["collect", "cluster", "percent", "random", "one", "language"])

    # Model
    parser.add_argument("--tokenizer", type=str, default='bert-base-uncased',
                        help="Pretrained BERT: bert-base-uncased, bert-base-multilingual-cased, xlm-roberta-base, etc.")
    parser.add_argument("--from_pretrained", type=str, default='bert-base-uncased',
                        help="Pretrained BERT: bert-base-uncased, bert-base-multilingual-cased, xlm-roberta-base, etc.,"
                             "Or full path to our pretrained model.")
    parser.add_argument("--base_lang", type=str, default="en",
                        help="Base language of the Creole. en or fr")
    # Logging
    parser.add_argument("--checkpoint_dir", type=str, default="")

    # Eval
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)

    #DRO stuff
    # Group
    parser.add_argument("--algo_log_metric", type=str, default="mse")
    parser.add_argument("--train_loader", type=str, default="standard", choices=['standard', 'group'])
    parser.add_argument("--uniform_over_groups", default=True, action="store_true")
    parser.add_argument("--n_groups_per_batch", type=int, default=1)
    parser.add_argument("--no_group_logging", default=True, action="store_true")
    parser.add_argument("--group_dro_step_size", type=float, default=0.01)
    # Training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--loss_function", type=str, default="cross_entropy",
                        choices=["cross_entropy", "mse", "multitask_bcd"])
    parser.add_argument("--num_epochs", type=int, default=1)
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


    return parser.parse_args()

def mask_dict_word(tokens, tokenizer, mlm_idxs):
    output_label = [-100] * len(tokens)
    for i in mlm_idxs:
        output_label[i] = tokens[i].item()
        tokens[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    return tokens.unsqueeze(0), torch.LongTensor(output_label).unsqueeze(0), mlm_idxs

def mask_1word(tokens, tokenizer):
    output_label = [-100] * len(tokens)
    rnd_token_ix = random.choice(np.arange(1, torch.where(tokens == 102)[0][0].item()))
    output_label[rnd_token_ix] = tokens[rnd_token_ix].item()
    tokens[rnd_token_ix] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    return tokens.unsqueeze(0), torch.LongTensor(output_label).unsqueeze(0), rnd_token_ix


def mask_allwords(tokens, tokenizer):
    max_ix = torch.where(tokens == 102)[0][0].item()
    batch_ids = torch.zeros(max_ix-1, tokens.size(0), dtype=torch.long)
    output_labels = torch.zeros_like(batch_ids) - 100
    mask_ix = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    for ix, tok_ix in enumerate(np.arange(1, max_ix)):
        sent = tokens.clone()
        out = tokens[ix+1].item()
        sent[ix+1] = mask_ix
        batch_ids[ix] = sent
        output_labels[ix, ix+1] = out
    return batch_ids, output_labels

def is_sublist(a, b, start):
    if not a: return True, "fuck"
    if not b: return False, "blah" 
    if b[:len(a)] == a:
        return start, len(a)
    else:
        return is_sublist(a, b[1:], start+1)  
    #return b[:len(a)] == a or is_sublist(a, b[1:])

def get_model_at_epoch_evals(model, eval_dataset, experiment, args):
    # PPL
    print("Computing PLL...")
    csv_columns = ["experiment", "i_example", "runningPPL"]

    results_dir = "/science/image/nlp-datasets/creoles/pplresults"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    if "mixed" in experiment:
        results_file = f"mixed_{args.creole}.csv"
    elif "creoleonly" in experiment:
        results_file = f"creoleonly_{args.creole}.csv"
    else:
        results_file = f"results_{args.creole}.csv"
    full_path_to_results = os.path.join(results_dir, results_file)

    ppls = []
    for i in tqdm(range(len(eval_dataset))):
        # Get tokenized sentence
        sent = eval_dataset.__getitem__(i)
        # Mask all tokens
        tokens, output_labels = mask_allwords(sent, tokenizer)
        tokens = tokens.to(device)
        output_labels = output_labels.to(device)
        with torch.no_grad():
            result = model(tokens, token_type_ids=None, labels=output_labels)
        lm_loss = F.cross_entropy(result.logits.view(-1, tokenizer.vocab_size), output_labels.view(-1), reduction="sum")
        ppls.append(lm_loss.cpu().item())
        row = [experiment, len(ppls), round(np.mean(ppls), 4)]

        if not os.path.isfile(full_path_to_results):
            with open(full_path_to_results, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(csv_columns)
                writer.writerow(row)
        else:  # append results
            with open(full_path_to_results, 'a') as file:
                writer = csv.writer(file)
                writer.writerow(row)

    mean_ppl = np.mean(ppls)
    print(f"Mean PLL: {mean_ppl}")
    row = [experiment, "FINAL", round(mean_ppl,4)]
    with open(full_path_to_results, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(row)
 
    return round(mean_ppl,4)
    
def clean_checkpoint_name(filepath):
        return filepath.split("conll")[-1]

if __name__ == "__main__":
    args = parse_args()
    # log.debug(args)
    #log_level = logging.DEBUG if args.debug else logging.INFO
    #logzero.loglevel(log_level)
    #logzero.formatter(logzero.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

    # load BERT tokenizer
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, do_lower_case=('uncased' in args.from_pretrained))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # If there's a GPU available...
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    # Load evaluation data
    eval_dataset = CreoleJsonDataset(src_file=args.file_path, tokenizer=tokenizer, base_language=args.base_lang, creole_only=True)

    """
    if args.creole == "singlish":
        eval_dataset = SinglishUDDataset(src_dir=args.file_path, tokenizer=tokenizer)
        creole_dictionary = creole_dict_reader(path=args.dictionary_path)
    elif args.creole == "naija":
        creole_dictionary = creole_dict_reader(path=args.dictionary_path)
        if "SUD" in args.file_path:
            eval_dataset = NaijaUDDataset(src_dir=args.file_path, tokenizer=tokenizer)
        elif "masakhane" in args.file_path:
            eval_dataset = NaijaMasakhaneNERDataset(src_dir=args.file_path, tokenizer=tokenizer)
        else:
            raise NotImplementedError
    elif args.creole == "haitian":
        eval_dataset = HaitianEvalDatasets(src_dir=args.file_path, tokenizer=tokenizer)
        creole_dictionary = haitian_dict_reader(path=args.dictionary_path)

    else:
        print(f"please specify the argument --creole= from ['singlish', 'haitian', 'naija']")
        print(f"other creoles have not been implemented")
        raise NotImplementedError
    """

    if args.experiment == "baseline":
        model_dirs = os.listdir(args.checkpoint_dir)
        model_dirs = sorted([d for d in model_dirs if d.isnumeric() and d in ["100000"]])
        for epoch in model_dirs:
            full_path = os.path.join(args.checkpoint_dir, epoch)
            model = AutoModelForMaskedLM.from_pretrained(full_path)
            model.to(device)
            model.eval()
            final_ppl = get_model_at_epoch_evals(model, eval_dataset, full_path, args)

            del model

    if args.experiment == "dro":
        args.optimizer_kwargs = {'eps': 1e-8}
        args.scheduler_kwargs = {'num_warmup_steps': 0}
        vocab_size = tokenizer.vocab_size

        train_dataset = CreoleDatasetWILDS(eval_dataset, tokenizer, group_strategy="one",
                                           group_file="",
                                           creole=args.creole)  # this one has (x, y, metadata)
        train_grouper = CombinatorialGrouper(dataset=train_dataset, groupby_fields=train_dataset.metadata_fields)
        group_ids = train_grouper.metadata_to_group(train_dataset.metadata_array)

        batch_sampler = GroupSampler(
            group_ids=group_ids,
            batch_size=args.batch_size,
            n_groups_per_batch=train_grouper.n_groups,
            uniform_over_groups=False,  # was True
            distinct_groups=False)  # was True
        torch.set_printoptions(threshold=100)
        print(f"group_ids: {group_ids} | num groups: {train_grouper.n_groups}")
        print(f"size of groups: {group_ids.size()}")
        print(Counter(group_ids.tolist()))
        train_loader = DataLoader(train_dataset, shuffle=False, sampler=None, batch_sampler=batch_sampler,
                                  drop_last=False)

        print(f"metadata_array: {train_dataset.metadata_array}")
        train_g = train_grouper.metadata_to_group(train_dataset.metadata_array)
        is_group_in_train = get_counts(train_g, train_grouper.n_groups) > 0
        print(f"is_group_in_train: {is_group_in_train}")
        # init DRO algorithm
        base_model = AutoModelForMaskedLM.from_pretrained(args.from_pretrained).to(device)
        #print(f"base model: {base_model}")
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
            config=args,
            model=base_model,
            d_out=train_dataset.y_size,
            grouper=train_grouper,
            loss=losses[args.loss_function],  # cross_entropy
            metric=None,  # MSE(),  #
            n_train_steps=len(train_loader) * args.num_epochs,
            is_group_in_train=is_group_in_train)


        #dirLUT = {'bert-base-uncased': 'bert', 'bert-base-multilingual-cased': 'mbert', 'xlm-roberta-base': 'xlmr'}
        #path_to_checkpoint = f"{args.checkpoint_dir}/{dirLUT[args.tokenizer]}/{args.creole}"
        all_the_models = os.listdir(args.checkpoint_dir)
        selected_models = sorted([m for m in all_the_models if args.group_strategy in m])

        look_up_models = []
        for model in selected_models:
            if any(e in model for e in ["100000"]):
                look_up_models.append(model)
        print(f"look up models: {look_up_models}")
        for cached_model in look_up_models:
            epoch_number = 100000 #cached_model[-5]
            full_path_to_model = os.path.join(args.checkpoint_dir, cached_model)
            print(f"check path: {full_path_to_model}")
            algorithm, epoch = load(algorithm=algorithm, path=full_path_to_model, device=device)
            print(f"epoch: {epoch}")
            algorithm.to(device)
            algorithm.eval()
            #exit(333)
            model = algorithm.model
            final_ppl = get_model_at_epoch_evals(model, eval_dataset, full_path_to_model, args)
            del model

    if args.experiment == "pretrained":
        model = AutoModelForMaskedLM.from_pretrained(args.from_pretrained)
        model.to(device)
        model.eval()
        final_ppl = get_model_at_epoch_evals(model, eval_dataset, args.from_pretrained, args)
        del model







