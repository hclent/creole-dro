#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --job-name=drone-b-haitian
#SBATCH --output=dro-one-bert-haitian-all.txt
#SBATCh --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=10G

source $HOME/.bashrc
conda activate /science/image/nlp-datasets/creoles/env/creole
which python

cd $HOME/creole-dro
echo "MEDICAL"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/haitian/1600_medical_domain_sentences.ht \
 --creole=haitian --experiment=dro --group_strategy=one \
 --tokenizer=bert-base-uncased --from_pretrained=bert-base-uncased \
 --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/bert/haitian --batch_size=16 \
 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/haitian/haitian-dictionary.txt

echo "###############################################"
echo "NEWSWIRE"

python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/haitian/newswire-all.ht \
 --creole=haitian --experiment=dro --group_strategy=one \
 --tokenizer=bert-base-uncased --from_pretrained=bert-base-uncased \
 --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/bert/haitian --batch_size=4 \
 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/haitian/haitian-dictionary.txt

echo "###############################################"
echo "GLOSSARY"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/haitian/glossary-all-fix.ht \
 --creole=haitian --experiment=dro  --group_strategy=one \
 --tokenizer=bert-base-uncased --from_pretrained=bert-base-uncased \
 --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/bert/haitian --batch_size=16 \
 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/haitian/haitian-dictionary.txt

echo "* Done! "
