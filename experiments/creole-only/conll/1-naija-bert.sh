#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanx:1
#SBATCH --job-name=Onaija-bert
#SBATCH --output=naija-bert.txt
#SBATCh --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=10G

source $HOME/.bashrc
conda activate /science/image/nlp-datasets/creoles/env/creole
which python

cd $HOME/creole-dro

echo "BASELINE BERT NAIJA ONLY"
python baseline.py --file_path=/science/image/nlp-datasets/creoles/data/train/naija/naija_only_groups.json \
    --tokenizer=bert-base-uncased --from_pretrained=bert-base-uncased \
    --base_lang=en --creole=naija \
    --output_dir=/science/image/nlp-datasets/creoles/outputs --checkpoint_dir=/science/image/nlp-datasets/creoles/conll/creoleonly/baseline \
    --action=train --batch_size=16

echo "DRO BERT NAIJA ONE ONLY"
python experiment1.py --file_path=/science/image/nlp-datasets/creoles/data/train/naija/naija_only_groups \
    --tokenizer=bert-base-uncased --from_pretrained=bert-base-uncased \
    --base_lang=en --creole=naija --group_strategy=one \
    --output_dir=/science/image/nlp-datasets/creoles/outputs --checkpoint_dir=/science/image/nlp-datasets/creoles/conll/creoleonly/dro \
    --batch_size=16

echo "DRO BERT NAIJA RANDOM ONLY"
python experiment1.py --file_path=/science/image/nlp-datasets/creoles/data/train/naija/naija_only_groups \
    --tokenizer=bert-base-uncased --from_pretrained=bert-base-uncased \
    --base_lang=en --creole=naija --group_strategy=random \
    --output_dir=/science/image/nlp-datasets/creoles/outputs --checkpoint_dir=/science/image/nlp-datasets/creoles/conll/creoleonly/dro \
    --batch_size=16

echo "DRO BERT NAIJA COLLECT ONLY"
python experiment1.py --file_path=/science/image/nlp-datasets/creoles/data/train/naija/naija_only_groups \
    --tokenizer=bert-base-uncased --from_pretrained=bert-base-uncased \
    --base_lang=en --creole=naija --group_strategy=collect \
    --output_dir=/science/image/nlp-datasets/creoles/outputs --checkpoint_dir=/science/image/nlp-datasets/creoles/conll/creoleonly/dro \
    --batch_size=16