"""
Precompute groups for DRO
Fasttext code adapted from: https://amitness.com/2019/07/identify-text-language-python/
"""
import os
import json
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
import numpy as np
import fasttext
import torch
from transformers import AutoTokenizer
from datasets import CreoleDataset, SinglishSMSDataset

def test(path, sentences):
    with open(path, 'r') as infile:
        sentence_dict = json.loads(infile.read())

    clusters = []

    for i, sent in enumerate(sentences):
       cluster = sentence_dict[sent]["cluster"]
       clusters.append(cluster)

    assert len(clusters) == len(sentences)
    print(f"Passes the test!")


def load_creole_and_all(file, creole):
    sentences = []

    with open(file, "r", encoding="utf-8") as input_file:
        entries = json.load(input_file)  # list of dicts

    for subdict in entries:
        for sent, lang in subdict.items():
            if lang == creole:
                sentences.append(sent)

    return sentences


def main(src_dir, src_file, creole, out_file):
    """

    :param src_dir:
    :param src_file:
    :param creole:
    :param out_file:
    :return: [] list of dicts {"sent": {"en": .8, "zh": .1, "yue": .1}
    """

    top_langs_distribution = defaultdict(list)
    chosen_langs_distribution = defaultdict(list)



    full_src_path = os.path.join(src_dir, src_file)

    sentences = load_creole_and_all(full_src_path, creole)

    #init out json
    out_json = []

    # load fasttext mode
    pretrained_model_path = "/Users/plq360/Desktop/tmp/lid.176.bin"
    model = fasttext.load_model(pretrained_model_path)

    creole_LUT = {"singlish": ["en", "zh", "ms", "ta"],
                  "haitian": ["fr", "yo", "es"],
                  "naija": ["en", "yo", "pt"]}

    sub_language_keys = creole_LUT[creole]

    #Now get the language predictions
    print(f"* predicting the languages in the examples ... ")
    predictions = model.predict(sentences, k=-1)  # get ALL the predictions!
    langs, scores = predictions

    #Build the json dict
    for sent, lang_list, score in zip(sentences, langs, scores):
        sent_dict = {}
        lang_LUT = {}
        for i, lang in enumerate(lang_list):
            lang_LUT[lang.split("__")[-1]] = i

        #chosen ones
        for lang in sub_language_keys:
            try:
                index = lang_LUT[lang]
                lang_score = float(score[index])
                sent_dict[lang] = lang_score
                chosen_langs_distribution[lang].append(lang_score)
            except Exception:
                sent_dict[lang] = float(0)
                chosen_langs_distribution[lang].append(0)
        #top 5
        for i, lang in enumerate(lang_list):
            if i < 5:
                code = lang.split("__")[-1]
                try:
                    top_langs_distribution[code].append(score)
                except Exception:
                    top_langs_distribution[code].append(score)

            else:
                break


        out_json.append({sent: sent_dict}) # [sent] = sent_dict


    #print(chosen_langs_distribution)
    print("###############################")
    print(top_langs_distribution.keys())


    # print(f"NUM EXAMPLES: {len(out_json)}")
    # #Print the json to a file
    # new_file = os.path.join(src_dir, out_file)
    # with open(new_file, 'w', encoding="utf-8") as o:
    #     json.dump(out_json, o, indent=1)

    #test(new_file, sentences)


main("/Users/plq360/Desktop/data/creoledata/train/singlish", "singlish_and_all.train.json", "singlish", "singlish_only_groups.json")
#main("/Users/plq360/Desktop/data/creoledata/train/naija", "naija_and_all.train.json", "naija","naija_only_groups.json")
#main("/Users/plq360/Desktop/data/creoledata/train/haitian", "haitian_and_all.train.json", "haitian", "haitian_only_groups.json")


