#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --job-name=pre-b-haitian
#SBATCH --output=pretrained-mbert-haitian-all.txt
#SBATCh --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem=15G

source $HOME/.bashrc
conda activate /science/image/nlp-datasets/creoles/env/creole
which python

cd $HOME/creole-dro
echo "MEDICAL"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/haitian/1600_medical_domain_sentences.ht \
 --creole=haitian --experiment=pretrained --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/haitian/haitian-dictionary.txt \
 --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased

echo "###############################################"
echo "NEWSWIRE"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/haitian/newswire-all.ht \
 --creole=haitian --experiment=pretrained --batch_size=8 \
 --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/haitian/haitian-dictionary.txt

echo "###############################################"
echo "GLOSSARY"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/haitian/glossary-all-fix.ht \
 --creole=haitian --experiment=pretrained --batch_size=8 \
 --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/haitian/haitian-dictionary.txt

echo "* Done! "
