#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --job-name=dro-p-naija
#SBATCH --output=dro-percent-naija-all.txt
#SBATCh --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem=15G

source $HOME/.bashrc
conda activate /science/image/nlp-datasets/creoles/env/creole
which python

cd $HOME/creole-dro

############# NER ###################
echo "DRO percent BERT NER"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/naija/masakhane-ner-pcm \
 --creole=naija --experiment=dro --group_strategy=percent \
 --tokenizer=bert-base-uncased --from_pretrained=bert-base-uncased \
 --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/bert/naija --batch_size=8 \
 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/naija/naija-dictionary.txt

echo "DRO percent MBERT NER"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/naija/masakhane-ner-pcm \
 --creole=naija --experiment=dro --group_strategy=percent \
 --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased \
 --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/mbert/naija --batch_size=8 \
 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/naija/naija-dictionary.txt

############# UD ###################

echo "DRO percent BERT UD"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/naija/SUD_Naija-NSC \
 --creole=naija --experiment=dro --group_strategy=percent \
 --tokenizer=bert-base-uncased --from_pretrained=bert-base-uncased \
 --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/bert/naija --batch_size=8 \
 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/naija/naija-dictionary.txt

echo "DRO percent MBERT UD"

python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/naija/SUD_Naija-NSC \
 --creole=naija --experiment=dro --group_strategy=percent \
 --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased \
 --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/mbert/naija --batch_size=8 \
 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/naija/naija-dictionary.txt