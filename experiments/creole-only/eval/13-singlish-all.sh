#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --job-name=singlish-eval
#SBATCH --output=singlish-eval-all.txt
#SBATCh --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=10G

source $HOME/.bashrc
conda activate /science/image/nlp-datasets/creoles/env/creole
which python

cd $HOME/creole-dro

# pretrained bert
echo "PRETRAINED BERT: "

python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/singlish/ACL17_UD_dataset/treebank/gold_pos/ --creole=singlish --experiment=pretrained --batch_size=6 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/singlish/singlish-dictionary.txt --tokenizer=bert-base-uncased --from_pretrained=bert-base-uncased

# dro one bert
echo "DRO ONE"

python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/singlish/ACL17_UD_dataset/treebank/gold_pos/ --creole=singlish --experiment=dro --batch_size=6 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/singlish/singlish-dictionary.txt --tokenizer=bert-base-uncased --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/bert/singlish --group_strategy=one

# pretrained mbert
echo "PRETRAINED MBERT:"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/singlish/ACL17_UD_dataset/treebank/gold_pos/ --creole=singlish --experiment=pretrained --batch_size=6 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/singlish/singlish-dictionary.txt --tokenizer=bert-base-multilingual-cased --from_pretrained=bert-base-multilingual-cased

# baseline mbert
echo "BASELINE MBERT"

python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/singlish/ACL17_UD_dataset/treebank/gold_pos/ --creole=singlish --experiment=baseline --batch_size=6 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/singlish/singlish-dictionary.txt --tokenizer=bert-base-multilingual-cased --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/baselines/mbert/singlish

# dro one mbert
echo "DRO ONE MBERT"

python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/singlish/ACL17_UD_dataset/treebank/gold_pos/ --creole=singlish --experiment=dro --batch_size=6 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/singlish/singlish-dictionary.txt --tokenizer=bert-base-multilingual-cased --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/mbert/singlish --group_strategy=one --from_pretrained=bert-base-multilingual-cased

# dro collect mbert
echo "DRO COLLECT MBERT"
python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/eval/singlish/ACL17_UD_dataset/treebank/gold_pos/ --creole=singlish --experiment=dro --batch_size=6 --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/singlish/singlish-dictionary.txt --tokenizer=bert-base-multilingual-cased --checkpoint_dir=/science/image/nlp-datasets/creoles/checkpoints/dro/mbert/singlish --group_strategy=collect --from_pretrained=bert-base-multilingual-cased
