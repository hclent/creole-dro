#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --job-name=sing-mbert
#SBATCH --output=singlish-mbert.txt
#SBATCh --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=10G

source $HOME/.bashrc
conda activate /science/image/nlp-datasets/creoles/env/creole
which python

cd $HOME/creole-dro

echo "BASELINE BERT singlish"
python baseline.py --file_path=/science/image/nlp-datasets/creoles/data/train/singlish/singlish_and_all.train.json \
    --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased \
    --base_lang=en --creole=singlish \
    --output_dir=/science/image/nlp-datasets/creoles/outputs --checkpoint_dir=/science/image/nlp-datasets/creoles/conll/mixed/baseline \
    --action=train --batch_size=15

echo "DRO BERT singlish ONE"
python experiment1.py --file_path=/science/image/nlp-datasets/creoles/data/train/singlish/singlish_and_all.train.json \
    --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased \
    --base_lang=en --creole=singlish --group_strategy=one \
    --output_dir=/science/image/nlp-datasets/creoles/outputs --checkpoint_dir=/science/image/nlp-datasets/creoles/conll/mixed/dro \
    --batch_size=15

echo "DRO BERT singlish RANDOM"
python experiment1.py --file_path=/science/image/nlp-datasets/creoles/data/train/singlish/singlish_and_all.train.json \
    --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased \
    --base_lang=en --creole=singlish --group_strategy=random \
    --output_dir=/science/image/nlp-datasets/creoles/outputs --checkpoint_dir=/science/image/nlp-datasets/creoles/conll/mixed/dro \
    --batch_size=15

echo "DRO BERT singlish LANGUAGE"
python experiment1.py --file_path=/science/image/nlp-datasets/creoles/data/train/singlish/singlish_and_all.train.json \
    --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased \
    --base_lang=en --creole=singlish --group_strategy=language \
    --output_dir=/science/image/nlp-datasets/creoles/outputs --checkpoint_dir=/science/image/nlp-datasets/creoles/conll/mixed/dro \
    --batch_size=15