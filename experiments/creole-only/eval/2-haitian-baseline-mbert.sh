#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --job-name=base-m-haitian
#SBATCH --output=baseline-mbert-haitian-all.txt
#SBATCh --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=15G

source $HOME/.bashrc
conda activate /science/image/nlp-datasets/creoles/env/creole
which python

cd $HOME/creole-dro

echo "MEDICAL"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/haitian/1600_medical_domain_sentences.ht \
 --creole=haitian --experiment=baseline --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/haitian/haitian-dictionary.txt \
 --tokenizer=bert-base-multilingual-cased --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/baselines/mbert/haitian

echo "###############################################"
echo "NEWSWIRE"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/haitian/newswire-all.ht \
 --creole=haitian --experiment=baseline --batch_size=6 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/haitian/haitian-dictionary.txt \
 --tokenizer=bert-base-multilingual-cased --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/baselines/mbert/haitian

echo "###############################################"
echo "GLOSSARY"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/haitian/glossary-all-fix.ht \
 --creole=haitian --experiment=baseline --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/haitian/haitian-dictionary.txt \
 --tokenizer=bert-base-multilingual-cased --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/baselines/mbert/haitian

echo "* Done! "
