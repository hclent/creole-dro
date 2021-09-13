import os
import re
import string
import spacy
import json
import random
import pandas as pd
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments
from transformers import BertForMaskedLM, AdamW, BertConfig
from transformers.data.data_collator import DataCollatorForLanguageModeling

import en_core_web_sm

# ==================================================================================================================== #
#                                                      Data utils                                                      #
# ==================================================================================================================== #

def split_sents(base_language, text_lines):
    """
    :param base_language: "en" or "fr" for English or French
    :param text_lines: list['sentences', 'sentences']
    :return: list['sent', 'sent', 'sent']
    """

    sentences = []

    if base_language == "en":
        nlp = en_core_web_sm.load()  # spacy.load("en_cor_web_sm")
        for text in text_lines:
            doc = nlp(text.strip())
            [sentences.append(s.text) for s in doc.sents]

    if base_language == "fr":
        for text in text_lines:
            [sentences.append(s.strip()) for s in re.split("\s\.\s", text)]
            #for s in re.split("\s\.\s", text):
            #    clean_s = clean_sent(s)
    sentences = [s for s in sentences if s.strip() != ""]

    return sentences


def get_max_length(sentences, tokenizer):
    """
    Get the max length for padding sentences
    :param sentences: list of sentences
    :return: max length of sentences
    """
    max_len = 0
    for sent in sentences:
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
    print(f" Max sent length: {max_len}")
    return max_len


def tokenize_data(sentences, tokenizer, max_len):
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(sent, add_special_tokens=True, max_length=max_len, pad_to_max_length=True,
                                             return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    # Convert the lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])

    return input_ids, attention_masks


# ==================================================================================================================== #
#                                                       Baseline                                                       #
# ==================================================================================================================== #

class CreoleDataset(Dataset):
    def __init__(self, src_file, tokenizer, base_language="en", nexamples=-1, evaluate=False):
        with open(src_file, "r", encoding="utf-8") as input_file:
            entries = input_file.readlines()[:nexamples]

        self.base_language = base_language

        self.sentences = split_sents(base_language, entries)
        print(f"NUMBER OF SENTENCES: {len(self.sentences)}")
        self.max_len = get_max_length(self.sentences, tokenizer)
        input_ids, attention_masks = tokenize_data(self.sentences, tokenizer, self.max_len)

        self.examples = input_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])

class CreoleJsonDataset(Dataset):
    def __init__(self, src_file, tokenizer, base_language="en", nexamples=-1, evaluate=False, creole_only=False):
        with open(src_file, "r", encoding="utf-8") as input_file:
            entries = json.load(input_file)[:nexamples] #list of dicts

        self.base_language = base_language

        self.sentences = []
        for subdict in entries:
            for sent, lang in subdict.items():
                if not creole_only:
                    self.sentences.append(sent)
                else: #if we are only looking at creole-only (for evaluation) skip the other examples in the file.
                    if lang in ["singlish", "naija", "haitian"]:
                        self.sentences.append(sent)

        print(f"NUMBER OF SENTENCES: {len(self.sentences)}")
        self.max_len = 128 #get_max_length(self.sentences, tokenizer)
        input_ids, attention_masks = tokenize_data(self.sentences, tokenizer, self.max_len)

        self.examples = input_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])

class SinglishSMSDataset(Dataset):
    def __init__(self, src_file, tokenizer, base_language="en", nexamples=-1, evaluate=False):
        with open(src_file, "r") as input_file:
            corpus_json = json.load(input_file)

        self.base_language = base_language
        self.sentences = []
        for message in corpus_json['smsCorpus']['message']:
            sent = str(message['text']['$']).strip("\n")
            if "\n" not in sent:
                self.sentences.append(sent)
            #self.sentences.append(str(message['text']['$']))

        print(f"NUMBER OF SENTENCES: {len(self.sentences)}")
        self.max_len = get_max_length(self.sentences, tokenizer)
        input_ids, attention_masks = tokenize_data(self.sentences, tokenizer, self.max_len)

        self.examples = input_ids


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        #return torch.tensor(self.examples[i])
        #recommended by pytorch
        return self.examples[i].clone().detach()#.requires_grad_(True)

class NaijaUDDataset(Dataset):
    def __init__(self, src_dir, tokenizer):

        self.base_language = "en"
        self.sentences = []

        filenames = [f for f in os.listdir(src_dir) if f.endswith(".conllu")]
        for f in filenames:
            with open(os.path.join(src_dir, f), "r") as input:
                lines = input.readlines()
                for line in lines:
                    if line.startswith("# text_ortho"):
                        clean_line = line[15:].strip()  # take off '# text_ortho = ':
                        self.sentences.append(clean_line)

        #FIXME: can not evaluate perplexity if we load the whole dataset
        # extended_path = os.path.join(path, "non_gold")
        # more_filenames = [f for f in os.listdir(extended_path) if f.endswith(".conllu")]
        # for f in more_filenames:
        #     with open(os.path.join(extended_path, f), "r") as input:
        #         lines = input.readlines()
        #         for line in lines:
        #             if line.startswith("# text_ortho"):
        #                 clean_line = line[15:].strip()
        #                 sentences.append(clean_line)

        print(f"NUMBER OF SENTENCES: {len(self.sentences)}")
        self.max_len = get_max_length(self.sentences, tokenizer)
        print(f"MAX SENTENCE LENGTH: {self.max_len}")
        input_ids, attention_masks = tokenize_data(self.sentences, tokenizer, self.max_len)

        self.examples = input_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])

class SinglishUDDataset(Dataset):
    def __init__(self, src_dir, tokenizer):
        self.base_language = "en"
        self.sentences = []

        files = ["train.conll", "dev.conll", "test.conll"]

        for f in files:
            full_path = os.path.join(src_dir, f)
            with open(full_path, "r") as indata:
                lines = indata.readlines()

            stack = []
            for line in lines:
                if line != "\n":
                    elems = line.split("\t")
                    token = elems[1]
                    stack.append(token)
                if line == "\n":
                    sent = " ".join(stack)
                    self.sentences.append(sent)
                    stack = []
        print(f"NUMBER OF SENTENCES: {len(self.sentences)}")
        self.max_len = get_max_length(self.sentences, tokenizer)
        print(f"MAX SENTENCE LENGTH: {self.max_len}")
        input_ids, attention_masks = tokenize_data(self.sentences, tokenizer, self.max_len)

        self.examples = input_ids
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])


class HaitianEvalDatasets(Dataset):
    def __init__(self, src_dir, tokenizer):
        self.base_language = "fr"
        self.sentences = []

        # files = ["1600_medical_domain_sentences.ht", "glossary-all-fix.ht", "newswire-all.ht"]
        # for f in files:
        #     full_path = os.path.join(src_dir, f)
        #     with open(full_path, "r") as indata:
        #         lines = indata.readlines()
        #         for line in lines: #clean out examples length 1?
        #             if len(line.split(" ")) > 1:
        #                 self.sentences.append(line.strip("\n")) #already in sentences

        #evaluating haitian datasets sepperately ... >_>
        with open(src_dir, "r") as indata:
            lines = indata.readlines()
            for line in lines: #clean out examples length 1?
                if len(line.split(" ")) > 1:
                    self.sentences.append(line.strip("\n")) #already in sentences
        print(f"NUMBER OF SENTENCES: {len(self.sentences)}")
        self.max_len = get_max_length(self.sentences, tokenizer)
        print(f"MAX SENTENCE LENGTH: {self.max_len}")
        input_ids, attention_masks = tokenize_data(self.sentences, tokenizer, self.max_len)

        self.examples = input_ids
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])

class NaijaMasakhaneNERDataset(Dataset):
    def __init__(self, src_dir, tokenizer):
        self.base_language = "en"
        self.sentences = []

        files = ["train.txt", "dev.txt", "test.txt"]
        for f in files:
            full_path = os.path.join(src_dir, f)
            with open(full_path, "r") as indata:
                lines = indata.readlines()

            stack = []
            for l in lines:
                if l != "\n":
                    elems = l.split(" ")
                    token = elems[0]
                    if token not in ['""""']:
                        stack.append(token)
                if l == "\n":
                    self.sentences.append(" ".join(stack))
                    stack = []

        print(f"NUMBER OF SENTENCES: {len(self.sentences)}")
        self.max_len = get_max_length(self.sentences, tokenizer)
        print(f"MAX SENTENCE LENGTH: {self.max_len}")
        input_ids, attention_masks = tokenize_data(self.sentences, tokenizer, self.max_len)

        self.examples = input_ids
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])


# ==================================================================================================================== #
#                                                        WILDS                                                         #
# ==================================================================================================================== #
class CreoleDatasetWILDS(Dataset):
    def __init__(self, base_dataset, tokenizer, group_strategy, group_file, creole, evaluate: bool = False):

        self.base_language = base_dataset.base_language
        self.creole = creole
        self.sentences = base_dataset.sentences
        self.max_len = base_dataset.max_len
        self.examples = base_dataset.examples

        self.y_size = base_dataset.examples.shape[1]

        self.is_classification = False

        #Input: self.sentences
        #Output: {metadata_df, identity_vars, metadata_array, metadata_fields, metadata_map}
        if group_strategy=="collect":
            self.metadata_df, self.identity_vars, self.metadata_array, self.metadata_fields, self.metadata_map = self.collect(sentences=self.sentences, group_file=group_file, creole=self.creole)
        #elif group_strategy=="cluster":
        #    self.metadata_df, self.identity_vars, self.metadata_array, self.metadata_fields, self.metadata_map = self.cluster(sentences=self.sentences, group_file=group_file)
        #elif group_strategy=="percent":
        #    self.metadata_df, self.identity_vars, self.metadata_array, self.metadata_fields, self.metadata_map = self.percent_base(sentences=self.sentences,group_file=group_file, base_language=self.base_language)
        elif group_strategy=="language": #a non-naive implementation of "collect"
            self.metadata_df, self.identity_vars, self.metadata_array, self.metadata_fields, self.metadata_map = self.language(sentences=self.sentences, group_file=group_file, creole=self.creole)
        elif group_strategy=="random":
            self.metadata_df, self.identity_vars, self.metadata_array, self.metadata_fields, self.metadata_map = self.random_groups(sentences=self.sentences)
        elif group_strategy=="one":
            self.metadata_df, self.identity_vars, self.metadata_array, self.metadata_fields, self.metadata_map = self.one_group(sentences=self.sentences)
        else:
            print("The grouping strategy that you tried to use does not exist")
            raise NotImplementedError


        ##### extracting the y's, because WILDS-compatible data objects need this #####
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        temp_training_args = TrainingArguments(
            output_dir="./dummy",
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_gpu_train_batch_size=1,
            save_steps=10_000,
            save_total_limit=2,
            prediction_loss_only=True,
        )
        temp_trainer = Trainer(
            model=BertForMaskedLM.from_pretrained('bert-base-uncased'),
            args=temp_training_args,
            data_collator=data_collator,
            train_dataset=base_dataset
        )
        temp_loader = temp_trainer.get_train_dataloader()

        self.temp_xs = []
        self.temp_ys = []
        for batch in temp_loader:
            self.temp_xs.append(batch["input_ids"])
            self.temp_ys.append(batch["labels"])

        self.temp_xs = torch.cat(self.temp_xs)
        self.temp_ys = torch.cat(self.temp_ys)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        x = self.temp_xs[i]
        y = self.temp_ys[i]
        metadata = self.metadata_array[i]
        return x, y, metadata

    def collect(self, sentences, group_file, creole):
        creole_LUT = {"singlish": ["en", "zh", "ms", "ta"],
                      "haitian": ["fr", "yo", "es"],
                      "naija": ["en", "yo", "pt"]}

        sub_language_keys = creole_LUT[creole]
        columns = ["x"] + sub_language_keys

        dfdata_default = defaultdict(list)

        with open(group_file, 'r', encoding="utf-8") as input_file:
            sentence_dict = json.loads(input_file.read())

        for sent, sent_dict in zip(sentences, sentence_dict):
            dfdata_default["x"].append(sent)
            for s, language_dict in sent_dict.items():
            #print(f"sent_dict is {sent_dict} (type({type(sent_dict)}))")
            #print(f"sent: {sent} is type {type(sent)}")
            #language_dict = sentence_dict[sent]
            #dfdata_default["x"].append(sent)
                assert sent == s

                for sub in sub_language_keys:
                    score = language_dict[sub]
                    dfdata_default[sub].append(score)

        dfdata = dict(dfdata_default)

        metadata_df = pd.DataFrame(dfdata, columns=columns)
        print(metadata_df.head)

        identity_vars = sub_language_keys
        metadata_fields = sub_language_keys

        metadata_array = torch.LongTensor((metadata_df.loc[:, identity_vars] >= .001).values) #if its at least 1%, then yes. otherwise no. 

        metadata_map = {} #so... I could get all the unique values and say 0=no, and anything else=yes
        #to save for space, maybe we wanna round to 2 decimals :/ No, because it looks like the predictions are soooo low shewlkarj.
        for lang in sub_language_keys:
            metadata_map[lang] = ["percent"]
            #metadata_map[lang] = ["true" for i in list(set(dfdata[lang])) if i > 0 else 1] <--- Nervous we MIGHT need something like this?
            #TODO: debug with grouper.py to see if this metadata_map is important
        return metadata_df, identity_vars, metadata_array, metadata_fields, metadata_map

    def language(self, sentences, group_file, creole):
        creole_LUT = {"singlish": ["singlish", "en", "zh", "ta", "ms"], #TODO: is this too many groups?
                      "haitian": ["haitian", "fr", "yo", "es"],
                      "naija": ["naija", "en", "yo", "pt"]}

        sub_language_keys = creole_LUT[creole]

        columns = ["x", "language"]
        dfdata= {"x": [], "language": []}

        with open(group_file, 'r', encoding="utf-8") as input_file:
            sentence_dict = json.loads(input_file.read())

	
        for sent, sent_dict in zip(sentences, sentence_dict):
            language = sent_dict[sent] #FIXME this is kludgy but we want to make sure the sentences are in the same order so if this fails we'll know
            dfdata["x"].append(sent)
            idx = sub_language_keys.index(language) #+ 1
            dfdata["language"].append(idx)

        metadata_df = pd.DataFrame(dfdata, columns=columns)
        print(metadata_df.head)

        identity_vars = columns[1:]
        metadata_fields = columns[1:]

        metadata_array = torch.LongTensor((metadata_df.loc[:, identity_vars]).values)
        sub_language_idxs = [i for i in range(0, len(sub_language_keys)+1)]
        metadata_map = {"language": sub_language_keys}

        return metadata_df, identity_vars, metadata_array, metadata_fields, metadata_map

    def cluster(self, sentences, group_file):

        columns = ["x", "cluster"]
        dfdata= {"x": [], "cluster": []}

        with open(group_file, 'r', encoding="utf-8") as input_file:
            sentence_dict = json.loads(input_file.read())

        for i, sent in enumerate(sentences):
            dfdata["x"].append(sent)
            dfdata["cluster"].append(sentence_dict[sent]["cluster"])

        metadata_df = pd.DataFrame(dfdata, columns=columns)
        print(metadata_df.head)

        identity_vars = columns[1:]
        metadata_fields = columns[1:]

        metadata_array = torch.LongTensor((metadata_df.loc[:, identity_vars]).values)
        metadata_map = {"cluster": ["0", "1", "2", "3", "4", "5"]}

        return metadata_df, identity_vars, metadata_array, metadata_fields, metadata_map

    def percent_base(self, sentences, group_file, base_language):
        columns = ["x", base_language]

        # init dfdata
        #dfdata = {"x": sentences, base_language: []}

        dfdata_default = defaultdict(list)

        with open(group_file, 'r', encoding="utf-8") as input_file:
            sentence_dict = json.loads(input_file.read())

        for sent in sentences:
            language_dict = sentence_dict[sent]
            dfdata_default["x"].append(sent)
            dfdata_default[base_language].append(language_dict[base_language])

        dfdata = dict(dfdata_default)

        metadata_df = pd.DataFrame(dfdata, columns=columns)
        print(metadata_df.head)

        identity_vars = [base_language]
        metadata_fields = [base_language]

        metadata_array = torch.LongTensor((metadata_df.loc[:, identity_vars] >= .001).values)

        metadata_map = {base_language: ["percent"]} #TODO: also need to see if this metadata_map is problematic

        return metadata_df, identity_vars, metadata_array, metadata_fields, metadata_map

    def random_groups(self, sentences):
        #Here is a random example of Grouping

        #### making up metadata #####
        dfdata = {'x': sentences,
                  'group1': [random.choice([0, 1]) for r in range(0, len(sentences))],
                  'group2': [random.choice([0, 1]) for r in range(0, len(sentences))]}

        metadata_df = pd.DataFrame(dfdata, columns=['x', 'group1', 'group2'])
        print(metadata_df.head)

        # identity vars need to be present in the df
        identity_vars = [
            'group1',
            'group2'
        ]

        metadata_array = torch.LongTensor((metadata_df.loc[:, identity_vars] >= 0.5).values)
        metadata_fields = ['group1', 'group2']
        metadata_map = {'group1': ['no', 'yes'],
                             'group2': ['no', 'yes']}  # 0=no, 1=yes. What the number values for the metadata map to

        return metadata_df, identity_vars, metadata_array, metadata_fields, metadata_map

    def one_group(self, sentences):
        #Assign everything the same thing, for a DRO baseline.

        #### making up metadata #####
        dfdata = {'x': sentences,
                  'group1': [0 for r in range(0, len(sentences))]}

        metadata_df = pd.DataFrame(dfdata, columns=['x', 'group1'])
        print(metadata_df.head)

        # identity vars need to be present in the df
        identity_vars = ['group1']

        metadata_array = torch.LongTensor((metadata_df.loc[:, identity_vars] >= 0.5).values)
        metadata_fields = ['group1']
        metadata_map = {'group1': ['yes']}  # 0=no, 1=yes. What the number values for the metadata map to

        return metadata_df, identity_vars, metadata_array, metadata_fields, metadata_map


