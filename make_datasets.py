import json
import spacy
import re
import os
import random
import csv
import torch
from transformers import AutoTokenizer
from datasets import CreoleDataset, SinglishSMSDataset

def split_sents(base_language, text_lines):
    """
    :param base_language: "en" or "fr" for English or French
    :param text_lines: list['sentences', 'sentences']
    :return: list['sent', 'sent', 'sent']
    """

    sentences = []

    if base_language == "en":
        nlp = spacy.load("en_core_web_sm") #spacy.load("en_cor_web_sm")
        for text in text_lines:
            doc = nlp(text.strip())
            [sentences.append(s.text) for s in doc.sents]

    #TODO: still relevant for new Haitian dataset?
    #
    # elif base_language == "fr":
    #     for text in text_lines:
    #         [sentences.append(s.strip()) for s in re.split("\s\.\s", text)]
    #         #for s in re.split("\s\.\s", text):
    #         #    clean_s = clean_sent(s)
    # sentences = [s for s in sentences if s.strip() != ""]

    else:
        nlp = spacy.load("xx_sent_ud_sm")
        for text in text_lines:
            text = text.strip()
            text = text.replace('\n', '')

            doc = nlp(text)
            [sentences.append(s.text) for s in doc.sents]

    return sentences


def load_other(file):
    path_to_other = "/Users/plq360/Desktop/data/creoledata/train/other"
    full_path = os.path.join(path_to_other, file)
    if file.endswith("json"):
        with open(full_path, "r") as injson:
            base_language = "ms"
            data = json.load(injson) #data[i]["text"] to get the text, and check that "language" = 'malay'
            malay_data = [d["text"] for d in data if d["language"] == "malay"]
            return malay_data, base_language
    else:
        with open(full_path, "r") as infile:
            base_language = file[-2:]
            data = infile.readlines()
            return data, base_language


def make_singlish(tokenizer):
    new_train = []
    new_dev = []

    new_dataset = []

    #load creole dataset
    #Use the right Dataset object
    path_to_singlish = "/Users/plq360/Desktop/data/creoledata/train/singlish"
    src_file ="smsCorpus_en_2015.03.09_all.json"
    full_src_path = os.path.join(path_to_singlish, src_file)
    dataset = SinglishSMSDataset(src_file=full_src_path, tokenizer=tokenizer, base_language="en")

    sentences = dataset.sentences

    for sent in sentences[:100]:
        print(sent)
    print("#############################################")

    #clean out some trash
    for i, s in enumerate(sentences):
        if "\n" not in s:
            new_dataset.append({s: "singlish"})

    random.shuffle(new_dataset)

    num_datapoints = len(new_dataset)
    num_train = int(num_datapoints * .95)
    num_dev = num_datapoints - num_train
    print(f"[SINGLISH]: Num total: {num_datapoints} ||| Num train: {num_train} ||| num_dev {num_dev}")

    #assign parts of new_dataset to train or dev
    [new_train.append(s) for s in new_dataset[:num_train]]
    [new_dev.append(s) for s in new_dataset[num_train:]]

    print(f"CONFIRM SINGLISH-ONLY SPLIT: len new_train: {len(new_train)} ||| len new_dev: {len(new_dev)}")

    #load other datasets
    other_datasets = ["news.en", "news.zh", "news.ta", "news-30k-ms.json"]
    for file in other_datasets:
        data, base_language = load_other(file)
        other_sents = split_sents(base_language, data)
        random.shuffle(other_sents)
        print(f"Len {base_language} sents: {len(other_sents)}")
        if len(other_sents) > num_datapoints:
            other_train = [{s: base_language} for s in other_sents[:num_train]]
            other_dev = [{s: base_language} for s in other_sents[num_train:num_train+num_dev]]
            print(f"{base_language}: train({len(other_train)}) ||| dev({len(other_dev)})")
            [new_train.append(s) for s in other_train]
            [new_dev.append(s) for s in other_dev]
        else: #split 95-5
            sub_train = len(other_sents) * .95
            other_train = [{s: base_language} for s in other_sents[:sub_train]]
            other_dev = [{s: base_language} for s in other_sents[sub_train:]]
            print(f"{base_language}: train({len(other_train)}) ||| dev({len(other_dev)})")
            [new_train.append(s) for s in other_train]
            [new_dev.append(s) for s in other_dev]

    #confirm numbers of stuff:
    print(f"LEN NEW TRAIN: {len(new_train)} ||| LEN NEW DEV {len(new_dev)}")
    print(f"Outputs... ")


    random.shuffle(new_train)
    random.shuffle(new_dev)

    #Print new TRAIN
    new_file = os.path.join(path_to_singlish, "singlish_and_all.train.json")
    with open(new_file, 'w', encoding="utf-8") as o:
        json.dump(new_train, o, indent=0)

    #Print new DEV
    new_file = os.path.join(path_to_singlish, "singlish_and_all.dev.json")
    with open(new_file, 'w', encoding="utf-8") as o:
        json.dump(new_dev, o, indent=0)



def make_haitian():
    new_train = []
    new_dev = []

    #load haitian train and dev
    path_to_haitian = "/Users/plq360/Desktop/data/creoledata/train/haitian"
    train_path = "disaster_response_messages_training.csv"

    with open(os.path.join(path_to_haitian, train_path), "r") as csvfile:
        rows = csv.reader(csvfile, delimiter=",", quotechar='"')
        for r in rows:
            h = r[3]
            if h!= '':
                new_train.append({h:"haitian"})



    dev_path = "disaster_response_messages_validation.csv"
    with open(os.path.join(path_to_haitian, dev_path), "r") as csvfile:
        rows = csv.reader(csvfile, delimiter=",", quotechar='"')
        for r in rows:
            h = r[3]
            if h!= '':
                new_dev.append({h:"haitian"})

    num_train = len(new_train)
    num_dev = len(new_dev)
    print(f"num train: {num_train}")
    print(f"num_dev: {num_dev}")

    #load other datasets
    other_datasets = ["news.fr", "news.yo", "news.es"]
    for file in other_datasets:
        data, base_language = load_other(file)
        other_sents = split_sents(base_language, data)
        random.shuffle(other_sents)
        print(f"Len {base_language} sents: {len(other_sents)}")
        if len(other_sents) > num_train:
            other_train = [{s: base_language} for s in other_sents[:num_train]]
            other_dev = [{s: base_language} for s in other_sents[num_train:num_train + num_dev]]
            print(f"{base_language}: train({len(other_train)}) ||| dev({len(other_dev)})")
            [new_train.append(s) for s in other_train]
            [new_dev.append(s) for s in other_dev]
        else:  # split 95-5
            sub_train = int(len(other_sents) * .95)
            other_train = [{s: base_language} for s in other_sents[:sub_train]]
            other_dev = [{s: base_language} for s in other_sents[sub_train:]]
            print(f"{base_language}: train({len(other_train)}) ||| dev({len(other_dev)})")
            [new_train.append(s) for s in other_train]
            [new_dev.append(s) for s in other_dev]

    # confirm numbers of stuff:
    print(f"LEN NEW TRAIN: {len(new_train)} ||| LEN NEW DEV {len(new_dev)}")
    print(f"Outputs... ")

    random.shuffle(new_train)
    random.shuffle(new_dev)

    # Print new TRAIN

    new_file = os.path.join(path_to_haitian, "haitian_and_all.train.json")
    with open(new_file, 'w', encoding="utf-8") as o:
        json.dump(new_train, o, indent=0)

    # Print new DEV
    new_file = os.path.join(path_to_haitian, "haitian_and_all.dev.json")
    with open(new_file, 'w', encoding="utf-8") as o:
        json.dump(new_dev, o, indent=0)
    pass

def make_naija(tokenizer):
    new_train = []
    new_dev = []

    path_to_naija = "/Users/plq360/Desktop/data/creoledata/train/naija"
    src_file ="pidgin_corpus.txt"
    full_src_path = os.path.join(path_to_naija, src_file)
    #load creole dataset
    dataset = CreoleDataset(src_file=full_src_path, tokenizer=tokenizer, base_language="en")

    sentences = dataset.sentences

    #load other datasets
    random.shuffle(sentences)

    num_datapoints = len(sentences)
    num_train = int(num_datapoints * .95)
    num_dev = num_datapoints - num_train
    print(f"[NAIJA]: Num total: {num_datapoints} ||| Num train: {num_train} ||| num_dev {num_dev}")

    # assign parts of new_dataset to train or dev
    [new_train.append({s: "naija"}) for s in sentences[:num_train]]
    [new_dev.append({s: "naija"}) for s in sentences[num_train:]]

    print(f"CONFIRM NAIJA-ONLY SPLIT: len new_train: {len(new_train)} ||| len new_dev: {len(new_dev)}")

    # load other datasets
    other_datasets = ["news.en", "news.yo", "news.pt"]
    for file in other_datasets:
        data, base_language = load_other(file)
        other_sents = split_sents(base_language, data)
        random.shuffle(other_sents)
        print(f"Len {base_language} sents: {len(other_sents)}")
        if len(other_sents) > num_datapoints:
            other_train = [{s: base_language} for s in other_sents[:num_train]]
            other_dev = [{s: base_language} for s in other_sents[num_train:num_train + num_dev]]
            print(f"{base_language}: train({len(other_train)}) ||| dev({len(other_dev)})")
            [new_train.append(s) for s in other_train]
            [new_dev.append(s) for s in other_dev]
        else:  # split 95-5
            sub_train = int(len(other_sents) * .95)
            other_train = [{s: base_language} for s in other_sents[:sub_train]]
            other_dev = [{s: base_language} for s in other_sents[sub_train:]]
            print(f"{base_language}: train({len(other_train)}) ||| dev({len(other_dev)})")
            [new_train.append(s) for s in other_train]
            [new_dev.append(s) for s in other_dev]

    # confirm numbers of stuff:
    print(f"LEN NEW TRAIN: {len(new_train)} ||| LEN NEW DEV {len(new_dev)}")
    print(f"Outputs... ")

    random.shuffle(new_train)
    random.shuffle(new_dev)

    # Print new TRAIN
    new_file = os.path.join(path_to_naija, "naija_and_all.train.json")
    with open(new_file, 'w', encoding="utf-8") as o:
        json.dump(new_train, o, indent=0)

    # Print new DEV
    new_file = os.path.join(path_to_naija, "naija_and_all.dev.json")
    with open(new_file, 'w', encoding="utf-8") as o:
        json.dump(new_dev, o, indent=0)


def main():
    #do everything on cpu
    #torch.device("cpu")
    #tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    #make_singlish(tokenizer)
    #make_naija(tokenizer)
    make_haitian()


main()