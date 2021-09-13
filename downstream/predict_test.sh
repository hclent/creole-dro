#!/bin/bash
#SBATCH --account=project_2000509
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=01:00:00

conda activate machamp
module load gcc/9.1.0

codebase=/projappl/project_2000509/machamp
loggingdir=/scratch/project_2000509/mdelhoneux/creole-dro

bert=$1
algo=$2
lang=$3
task=$4
sample=$5

#assumes only one
serialized_dir_a=`ls $loggingdir/logs/$sample-$lang-$task-$bert-$algo/`
serialized_dir=$loggingdir/logs/$sample-$lang-$task-$bert-$algo/$serialized_dir_a
if [[ $task == "upos" ]]
then
    if [[ $lang == "singlish" ]]
        then
            input_file='data/eval/singlish/ACL17_UD_dataset/treebank/gold_pos/test.conll'
        else
            input_file='data/UD_Naija-NSC/pcm_nsc-ud-test.conllu'
        fi
    output_file=$serialized_dir/test.pred.conll
else
    input_file='data/eval/naija/masakhane-ner-pcm/test.txt'
    output_file=$serialized_dir/test.pred.txt
fi

python $codebase/predict.py $serialized_dir/model.tar.gz $input_file $output_file --device 0 
