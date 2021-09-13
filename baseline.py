#!/usr/bin/env python3

"""Usage: baseline.py ACTION BASE FILE [options]

Finetune BERT on creole data



Options:
  -P, --path-to-model      The path to the directory containing your pytorch_model.bin

TODO:
  -C, --checkpoint FILE    Continue from this checkpoint .torch file.
  -d, --device DEVICE      Run on this device. [default: cpu]
  --no-checkpoint          Disable saving of intermediate checkpoints.
  --no-save                Disable model & checkpoint saving.

Example:
  python baseline.py train en pidgin_corpus.txt --debug
  python baseline.py evaluate en other_corpus.txt -P ./naija-1/pytorch_model.bin

  python baseline.py train fr SMS-train-raw.ht --debug

"""

import logging
import logzero
import argparse

import time
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForMaskedLM, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import pipeline

from utils import format_time
from datasets import CreoleJsonDataset, CreoleDataset, NaijaUDDataset, SinglishSMSDataset


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--file_path", type=str, default="",
                        help="Path to the data you are trying to finetune on or evaluate from")
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
    # Logging
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--debug", default=False, action="store_true",
                        help="Enable debug-level logging.")
    # Training
    parser.add_argument("--action", type=str, default="train", choices=["train", "evaluate"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Former default was 5e-5")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                        help="Default in run_glue.py")
    # Evaluation
    parser.add_argument("--eval_batch_size", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # log.debug(args)
    log_level = logging.DEBUG if args.debug else logging.INFO
    logzero.loglevel(log_level)
    logzero.formatter(logzero.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

    # load BERT tokenizer
    print('Loading BERT tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, do_lower_case=('uncased' in args.from_pretrained))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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

    if args.action == "train":

        if args.creole == "singlish":
            train_dataset = CreoleJsonDataset(src_file=args.file_path, tokenizer=tokenizer, base_language="en", creole_only=args.creole_only)

        elif args.creole == "naija":
            train_dataset = CreoleJsonDataset(src_file=args.file_path, tokenizer=tokenizer, base_language="en", creole_only=args.creole_only)

        elif args.creole == "haitian":
            train_dataset = CreoleJsonDataset(src_file=args.file_path, tokenizer=tokenizer, base_language="fr", creole_only=args.creole_only)

        else:
            print(f"please specify the argument --creole= from ['singlish', 'haitian', 'naija']")
            print(f"other creoles have not been implemented")
            raise NotImplementedError

        """
        CREOLE-ONLY daataset
        
        if args.creole == "singlish":
            train_dataset = SinglishSMSDataset(src_file=args.file_path, tokenizer=tokenizer, base_language="en")

        elif args.creole == "naija":
            train_dataset = CreoleDataset(src_file=args.file_path, tokenizer=tokenizer, base_language="en")

        elif args.creole == "haitian":
            train_dataset = CreoleDataset(src_file=args.file_path, tokenizer=tokenizer, base_language="fr")

        else:
            print(f"please specify the argument --creole= from ['singlish', 'haitian', 'naija']")
            print(f"other creoles have not been implemented")
            raise NotImplementedError
        """

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.num_epochs,
            per_gpu_train_batch_size=args.batch_size,
            save_steps=20_000,
            save_total_limit=2,
            prediction_loss_only=True,
        )

        model = AutoModelForMaskedLM.from_pretrained(args.from_pretrained)
        model.to(device)

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        train_loader = trainer.get_train_dataloader()  # has "input_ids" and "labels"

        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=0.01)
        print(f"OPTIMIZER: {optimizer}")
        total_steps = 100000 #len(train_loader) * args.num_epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.num_warmup_steps,
                                                    num_training_steps=total_steps)

        dirLUT = {'bert-base-uncased': 'bert', 'bert-base-multilingual-cased': 'mbert', 'xlm-roberta-base': 'xlmr',
              'prajjwal1/bert-tiny': 'tinybert', 'prajjwal1/bert-small': 'smallbert'}


        steps_so_far = 0
        for epoch_i in range(0, args.num_epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.num_epochs))
            print('Training...')
            t0 = time.time()
            total_train_loss = 0
            model.train()
            for step, batch in enumerate(train_loader):
                if step % 100 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))
                b_input_ids = batch["input_ids"].to(device)
                b_labels = batch["labels"].to(device)
                model.zero_grad()
                result = model(b_input_ids, token_type_ids=None, labels=b_labels, return_dict=True)
                loss = result.loss
                logits = result.logits
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                steps_so_far += 1
                if steps_so_far % 20000 == 0 and not steps_so_far == 0:
                    print(f"Saving model @ epoch {epoch_i} || total steps {steps_so_far}")
                    path_to_checkpoint = f"{args.checkpoint_dir}/{dirLUT[args.tokenizer]}/{args.creole}/{steps_so_far}"
                    Path(path_to_checkpoint).mkdir(parents=True, exist_ok=True)
                    trainer.save_model(path_to_checkpoint)
                if steps_so_far == 100000:
                    print(f"Saving model @ epoch {epoch_i} || total steps {steps_so_far} || END")
                    path_to_checkpoint = f"{args.checkpoint_dir}/{dirLUT[args.tokenizer]}/{args.creole}/{steps_so_far}"
                    Path(path_to_checkpoint).mkdir(parents=True, exist_ok=True)
                    trainer.save_model(path_to_checkpoint)
                    print("Saved Model. EXITING TRAINING ...")
                    exit(100000)

            avg_train_loss = total_train_loss / len(train_loader)
            training_time = format_time(time.time() - t0)

        print(f"Done training! ")



"""

 
Useful links:
https://huggingface.co/transformers/model_doc/bert.html#bertformaskedlm
https://huggingface.co/blog/how-to-train


Useful tutorials:
https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX#scrollTo=6J-FYdx6nFE_

Check that it actually trained: 
https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb#scrollTo=YpvnFFmZJD-N

Perplexity:
https://huggingface.co/transformers/perplexity.html


"""




