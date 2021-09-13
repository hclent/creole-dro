import json
import os
import spacy
import random
import csv

def read_UD(file):
    sentences = []
    with open(file, "r") as input:
        lines = input.readlines()
        for line in lines:
            if line.startswith("# text"):
                clean_line = line[9:].strip()
                sentences.append(clean_line)
    return sentences

def read_NaijaUD(src_dir):
    naija_sentences = []
    english_sentences = []
    filenames = [f for f in os.listdir(src_dir) if f.endswith(".conllu")]
    for f in filenames:
        with open(os.path.join(src_dir, f), "r") as input:
            lines = input.readlines()
            for line in lines:
                if line.startswith("# text_ortho"):
                    clean_line = line[15:].strip()  # take off '# text_ortho = '
                    naija_sentences.append(clean_line)
                if line.startswith("# text_en"):
                    clean_line = line[12:].strip() # take off '# text_en = '
                    english_sentences.append(clean_line)
    return naija_sentences, english_sentences

def read_SinglishUD(full_path):
    sentences = []

    with open(full_path, "r") as indata:
        lines = indata.readlines()

    stack = []
    for line in lines:
        if line != "\n":
            elems = line.split("\t")
            token = elems[1]
            stack.append(token.strip())
        if line == "\n":
            sent = " ".join(stack)
            sentences.append(sent)
            stack = []
    return sentences

def read_HaitianExtra(full_path):
    sentences = []
    # evaluating haitian datasets sepperately ... >_>
    with open(full_path, "r") as indata:
        lines = indata.readlines()
        for line in lines:  # clean out examples length 1?
            if len(line.split(" ")) > 1:
                sentences.append(line.strip("\n").strip())  # already in sentences
    return sentences

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
            [sentences.append(s.text.strip()) for s in doc.sents]
    else:
        nlp = spacy.load("xx_sent_ud_sm")
        for text in text_lines:
            text = text.strip()
            text = text.replace('\n', '')

            doc = nlp(text)
            [sentences.append(s.text) for s in doc.sents]

    return sentences

def load_news(full_path):
    """
    for reading "news.LANG" WMT2020 files
    """
    with open(full_path, "r") as infile:
        base_language = full_path[-2:]
        data = infile.readlines()
        sentences = split_sents(base_language, data)
        return sentences

def readJson(file, lang):
    sentences = []
    with open(file, "r") as injson:
        data = json.load(injson, encoding="utf-8")
        for lil_d in data:
            for sent, label in lil_d.items():
                if label == lang:
                    sentences.append(sent)
    return sentences

def makeVocab(list_of_sents, language="en"):
    vocab = set()
    vocab.add("<unk>")

    """ 
    #This is the better way to build the vocab, but the proxy-a-distance code just splits on white space
    
    if language in ["haitian", "naija", "singlish"]:
        nlp = spacy.load("xx_sent_ud_sm")
    elif language == "en":
        nlp = spacy.load("en_core_web_sm")  # spacy.load("en_cor_web_sm")
    elif language == "fr":
        nlp = spacy.load("fr_core_news_sm")
    else:
        print("Youve supplied a language that is not supported by the tokenizer")
        raise NotImplementedError

    tokenizer = nlp.tokenizer
    for sent in list_of_sents:
        tokens = tokenizer(sent)
        [vocab.add(t.text) for t in tokens if t.text!= " "]
    """

    for sent in list_of_sents:
        words = sent.split(" ")
        [vocab.add(w.strip()) for w in words if w != " "]


    return list(vocab)

def output_file(list_of_stuff, filename):
    out_dir = "/Users/plq360/Desktop/data/creoledata/proxydata"
    with open(os.path.join(out_dir, filename), "w") as output:
        for i, stuff in enumerate(list_of_stuff):
            if stuff != '':
                if i+1 != len(list_of_stuff):
                    output.write(f"{stuff}\n")
                else:
                    output.write(f"{stuff}")


def main():
    num_src = 3050

    domain_3_src = [] #parallel to Haitian-1
    domain_4_src = [] #parallel to Naija-2
    domain_5_src = [] #parallel to Haitian-2

    for language in ["naija", "singlish", "haitian", "en", "fr"]:

        if language == "naija":
            domain_1 = readJson("/Users/plq360/Desktop/data/creoledata/train/naija/naija_and_all.train.json", "naija")

            domain_1_src = domain_1[:num_src]

            domain_2_naija, domain_4_english = read_NaijaUD("/Users/plq360/Desktop/data/creoledata/eval/naija/SUD_Naija-NSC")
            combo = list(zip(domain_2_naija, domain_4_english))
            random.shuffle(combo)
            combo_clipped = combo[:num_src]

            domain_2_src = [p[0] for p in combo_clipped]
            domain_4_src = [p[1] for p in combo_clipped]

            all_sentences = domain_1_src + domain_2_src
            print(f"len all sentences: {len(all_sentences)}")

            vocab_list = makeVocab(list_of_sents=all_sentences, language="naija")


            print(f"Naija d1 src [pidgin corpus]: {len(domain_1_src)}")
            print(f"Naija d2 src [NUD]: {len(domain_2_src)}")
            print(f"English d4 src [NUD - parallel to Naija d1 NUD]: {len(domain_4_src)}")
            print(f"len vocab: {len(vocab_list)}")

            output_file(domain_1_src, "naija-corpus.src")
            output_file(domain_2_src, "naija-NUD.src")
            output_file(domain_4_src, "english-NUD.src")
            output_file(vocab_list, "naija.vocab")

        if language == "singlish":
            domain_1 = readJson("/Users/plq360/Desktop/data/creoledata/train/singlish/singlish_and_all.train.json", "singlish")
            domain_1_src = domain_1[:num_src]

            domain_2 = read_SinglishUD("/Users/plq360/Desktop/data/creoledata/eval/singlish/TALLIP19_UD_dataset/gold_pos/train.ext.conll")
            random.shuffle(domain_2)
            domain_2_src = domain_2[:num_src]

            print(f"Singlish d1 src [SMS]: {len(domain_1_src)}")
            print(f"Singlish d2 src [SUD]: {len(domain_2_src)}")

            all_sentences = domain_1_src + domain_2_src
            vocab_list = makeVocab(list_of_sents=all_sentences, language="singlish")
            print(f"len vocab: {len(vocab_list)}")

            output_file(domain_1_src, "singlish-SMS.src")
            output_file(domain_2_src, "singlish-SUD.src")
            output_file(vocab_list, "singlish.vocab")

        if language == "haitian":
            # domain_1 = readJson("/Users/plq360/Desktop/data/creoledata/train/haitian/haitian_and_all.train.json", "haitian")
            # domain_1_src = domain_1[:num_src]

            domain_1 = []
            domain_3 = [] #PARALLEL ENGLISH DOMAIN 3

            # load haitian train and dev
            path_to_haitian = "/Users/plq360/Desktop/data/creoledata/train/haitian"
            train_path = "disaster_response_messages_training.csv"

            with open(os.path.join(path_to_haitian, train_path), "r") as csvfile:
                rows = csv.reader(csvfile, delimiter=",", quotechar='"')
                for r in rows:
                    en = r[2]
                    h = r[3]
                    if h != '':
                        domain_1.append(h)
                    if en != '':
                        domain_3.append(en)

            domain_1_src = domain_1[:num_src]
            domain_3_src = domain_3[:num_src]


            domain_2 = read_HaitianExtra("/Users/plq360/Desktop/data/creoledata/eval/haitian/newswire-all.ht")
            domain_5 = read_HaitianExtra("/Users/plq360/Desktop/data/creoledata/eval/haitian/newswire-all.en")

            combo = list(zip(domain_2, domain_5))
            random.shuffle(combo)
            combo_clipped = combo[:num_src]

            domain_2_src = [p[0] for p in combo_clipped] #haitian
            domain_5_src = [p[1] for p in combo_clipped] #english

            print(f"Haitian d1 src [emergency]: {len(domain_1_src)}")
            print(f"Haitian d2 src [newswire]: {len(domain_2_src)}")
            print(f"English d3 src [parallel to Haitian emergency]: {len(domain_3_src)}")
            print(f"English d5 src [parallel to Haitain newswire]: {len(domain_5_src)}")

            all_sentences = domain_1_src + domain_2_src
            vocab_list = makeVocab(list_of_sents=all_sentences, language="haitian")
            print(f"len vocab: {len(vocab_list)}")

            output_file(domain_1_src, "haitian-emergency.src")
            output_file(domain_2_src, "haitian-newswire.src")
            output_file(domain_3_src, "english-emergency.src")
            output_file(domain_5_src, "english-newswire.src")
            output_file(vocab_list, "haitian.vocab")

        if language == "en":
            domain_1 = readJson("/Users/plq360/Desktop/data/creoledata/train/naija/naija_and_all.train.json", "en")
            domain_1_src = domain_1[:num_src]

            domain_2 = read_UD("/Users/plq360/Desktop/data/creoledata/train/other/en_ewt-ud-train.conllu")
            random.shuffle(domain_2)
            domain_2_src = domain_2[:num_src]

            all_sentences = domain_1_src + domain_2_src + domain_3_src + domain_4_src + domain_5_src
            vocab_list = makeVocab(list_of_sents=all_sentences, language="en")

            print(f"english d1 src [wmt-news]: {len(domain_1_src)}")
            print(f"english d2 src [ewt-UD]: {len(domain_2_src)}")
            print(f"len vocab: {len(vocab_list)}")

            output_file(domain_1_src, "english-wmt-news.src")
            output_file(domain_2_src, "english-ewt-UD.src")
            output_file(vocab_list, "english.vocab")

        if language == "fr":
            domain_1 = readJson("/Users/plq360/Desktop/data/creoledata/train/haitian/haitian_and_all.train.json", "fr")
            domain_1_src = domain_1[:num_src]

            domain_2 = read_UD("/Users/plq360/Desktop/data/creoledata/train/other/fr_gsd-ud-train.conllu")
            random.shuffle(domain_2)
            domain_2_src = domain_2[:num_src]

            all_sentences = domain_1_src + domain_2_src
            vocab_list = makeVocab(list_of_sents=all_sentences, language="fr")

            print(f"French d1 src [wmt-news]: {len(domain_1_src)}")
            print(f"French d2 src [french UD]: {len(domain_2_src)}")
            print(f"len vocab: {len(vocab_list)}")

            output_file(domain_1_src, "french-wmt-news.src")
            output_file(domain_2_src, "french-FUD.src")
            output_file(vocab_list, "french.vocab")



main()

"""
len all sentences: 6100
Naija d1 src [pidgin corpus]: 3050
Naija d2 src [NUD]: 3050
English d4 src [NUD - parallel to Naija d1 NUD]: 3050
len vocab: 13353
Singlish d1 src [SMS]: 3050
Singlish d2 src [SUD]: 3050
len vocab: 13839
Haitian d1 src [emergency]: 3050
Haitian d2 src [newswire]: 3050
English d3 src [parallel to Haitian emergency]: 3050
English d5 src [parallel to Haitain newswire]: 3050
len vocab: 20443
english d1 src [wmt-news]: 3050
english d2 src [ewt-UD]: 3050
len vocab: 38759
French d1 src [wmt-news]: 3050
French d2 src [french UD]: 3050
len vocab: 31491

"""