#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --job-name=drone-m-naija
#SBATCH --output=dro-one-mbert-naija.txt
#SBATCh --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
#SBATCH --mem=10G

source $HOME/.bashrc
conda activate /science/image/nlp-datasets/creoles/env/creole
which python

cd $HOME/creole-dro

python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/naija/masakhane-ner-pcm \
 --creole=naija --experiment=dro --group_strategy=one \
 --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased \
 --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/mbert/naija --batch_size=8

echo "###############################################"

python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/naija/SUD_Naija-NSC \
 --creole=naija --experiment=dro --group_strategy=one \
 --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased \
 --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/mbert/naija --batch_size=8



echo "* Done! "
