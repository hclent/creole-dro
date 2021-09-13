#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --job-name=dro-percent-singlish
#SBATCH --output=dro-percent-singlish.txt
#SBATCh --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=15G

source $HOME/.bashrc
conda activate /science/image/nlp-datasets/creoles/env/creole
which python

cd $HOME/creole-dro


#### Dro-percent BERT
echo "DRO percent BERT"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/singlish/ACL17_UD_dataset/treebank/gold_pos/ \
  --creole=singlish --experiment=dro --batch_size=6 \
  --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/singlish/singlish-dictionary.txt --tokenizer=bert-base-uncased \
  --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/bert/singlish --group_strategy=percent

#### Dro-percent MBERT
echo "DRO percent MBERT"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/singlish/ACL17_UD_dataset/treebank/gold_pos/ \
  --creole=singlish --experiment=dro --batch_size=6 \
  --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/singlish/singlish-dictionary.txt \
  --tokenizer=bert-base-multilingual-cased --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/mbert/singlish \
  --group_strategy=percent --from_pretrained=bert-base-multilingual-cased