#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanx:1
#SBATCH --job-name=ohait-mbert
#SBATCH --output=ohaitian-mbert.txt
#SBATCh --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=10G

source $HOME/.bashrc
conda activate /science/image/nlp-datasets/creoles/env/creole
which python

cd $HOME/creole-dro

echo "BASELINE mbert haitian"
python baseline.py --file_path=/science/image/nlp-datasets/creoles/data/train/haitian/haitian_only_groups.json \
    --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased \
    --base_lang=fr --creole=haitian \
    --output_dir=/science/image/nlp-datasets/creoles/outputs --checkpoint_dir=/science/image/nlp-datasets/creoles/conll/creoleonly/baseline \
    --action=train --batch_size=16

echo "DRO mbert haitian ONE"
python experiment1.py --file_path=/science/image/nlp-datasets/creoles/data/train/haitian/haitian_only_groups.json \
    --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased \
    --base_lang=fr --creole=haitian --group_strategy=one \
    --output_dir=/science/image/nlp-datasets/creoles/outputs --checkpoint_dir=/science/image/nlp-datasets/creoles/conll/creoleonly/dro \
    --batch_size=16

echo "DRO mbert haitian RANDOM"
python experiment1.py --file_path=/science/image/nlp-datasets/creoles/data/train/haitian/haitian_only_groups.json \
    --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased \
    --base_lang=fr --creole=haitian --group_strategy=random \
    --output_dir=/science/image/nlp-datasets/creoles/outputs --checkpoint_dir=/science/image/nlp-datasets/creoles/conll/creoleonly/dro \
    --batch_size=16

echo "DRO mbert haitian LANGUAGE"
python experiment1.py --file_path=/science/image/nlp-datasets/creoles/data/train/haitian/haitian_only_groups.json \
    --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased \
    --base_lang=fr --creole=haitian --group_strategy=language \
    --output_dir=/science/image/nlp-datasets/creoles/outputs --checkpoint_dir=/science/image/nlp-datasets/creoles/conll/creoleonly/dro \
    --batch_size=16