#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=5:00:00

conda activate machamp
module load gcc/9.1.0

codebase=/projappl/project_2000509/machamp
loggingdir=/scratch/project_2000509/mdelhoneux/creole-dro
downstreamdir=$loggingdir/downstream

cd $loggingdir

bert=$1 #bert or mbert
algo=$2 #baseline or dro
lang=$3 #naija or singlish
task=$4 #ner (naija) or upos (singlish or naija)
sample=$5 #mixed or creoleonly

if [[ $bert == "mbert" ]]
then
    bert_pretrained='bert-base-multilingual-cased'
else
    bert_pretrained='bert-base-uncased'
fi

if [[ $algo == "dro" ]]
then
    if [[ $sample == "creoleonly" ]]
    then
        bert_path=$loggingdir/$sample/dro/$bert/$lang/dro_$lang\_collect_100000.pth

    else
        bert_path=$loggingdir/$sample/dro/$bert/$lang/dro_$lang\_language_100000.pth
    fi
    parameters_config=$downstreamdir/configs/params_$task.json
else
    bert_pretrained=$loggingdir/$sample/baseline/$bert/$lang/100000/
    parameters_config=$downstreamdir/configs/params_$task\_b.json
fi

BERT_PATH=$bert_path \
BERT_PRETRAINED=$bert_pretrained \
CUDA_VISIBLE_DEVICES=0 \
CUDA_DEVICE=0 \
BATCH_SIZE=32 \
EPOCHS=10 \
python $codebase/train.py --parameters_config $parameters_config --dataset_config $downstreamdir/configs/$task\_$lang.json --name $sample-$lang-$task-$bert-$algo

