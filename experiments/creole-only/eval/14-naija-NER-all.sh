#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --job-name=naija-ner-eval
#SBATCH --output=naija-NER-all.txt
#SBATCh --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=10G

source $HOME/.bashrc
conda activate /science/image/nlp-datasets/creoles/env/creole
which python

cd $HOME/creole-dro

echo "PRETRAINED BERT"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/naija/masakhane-ner-pcm \
 --creole=naija --experiment=pretrained \
 --tokenizer=bert-base-uncased --from_pretrained=bert-base-uncased \
 --batch_size=8 \
 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/naija/naija-dictionary.txt

echo "###################"

echo "BASELINE BERT"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/naija/masakhane-ner-pcm \
 --creole=naija --experiment=baseline \
 --tokenizer=bert-base-uncased --from_pretrained=bert-base-uncased \
 --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/baselines/bert/naija --batch_size=8 \
 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/naija/naija-dictionary.txt

echo "###################"

echo "DRO ONE BERT"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/naija/masakhane-ner-pcm \
 --creole=naija --experiment=dro --group_strategy=one \
 --tokenizer=bert-base-uncased --from_pretrained=bert-base-uncased \
 --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/bert/naija --batch_size=8 \
 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/naija/naija-dictionary.txt

echo "###################"

echo "DRO COLLECT BERT"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/naija/masakhane-ner-pcm \
 --creole=naija --experiment=dro --group_strategy=collect \
 --tokenizer=bert-base-uncased --from_pretrained=bert-base-uncased \
 --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/bert/naija --batch_size=8 \
 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/naija/naija-dictionary.txt

echo "###################"

echo "PRETRAINED MBERT"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/naija/masakhane-ner-pcm \
 --creole=naija --experiment=pretrained \
 --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased \
 --batch_size=8 \
 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/naija/naija-dictionary.txt

echo "###################"
echo "BASELINE MBERT"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/naija/masakhane-ner-pcm \
 --creole=naija --experiment=baseline \
 --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased \
 --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/baselines/mbert/naija --batch_size=8 \
 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/naija/naija-dictionary.txt

echo "###################"
echo "DRO ONE MBERT"

python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/naija/masakhane-ner-pcm \
 --creole=naija --experiment=dro --group_strategy=one \
 --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased \
 --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/mbert/naija --batch_size=8 \
 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/naija/naija-dictionary.txt

echo "###################"
echo "DRO COLLECTION MBERT"

python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/naija/masakhane-ner-pcm \
 --creole=naija --experiment=dro --group_strategy=collect \
 --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased \
 --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/mbert/naija --batch_size=8 \
 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/naija/naija-dictionary.txt

