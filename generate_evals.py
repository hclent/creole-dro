import os

def writeSH(out_file_name, head, experiment, creole, strategy, script):
    out_file_dir = "/science/image/nlp-datasets/creoles/results/runppl"
    full_path = os.path.join(out_file_dir, out_file_name)

    first_print = [ "#!/bin/bash",
                    "#SBATCH --partition=image1",
                    f"#SBATCH --job-name={creole}-{experiment}-{strategy}",
                    f"#SBATCH --output={head}-{creole}-{experiment}-{strategy}.txt",
                    "#SBATCh --nodes=1",
                    "#SBATCH --ntasks=1",
                    "#SBATCH --cpus-per-task=4",
                    "#SBATCH --time=1-00:00:00",
                    "#SBATCH --mem=20G",
                    "source $HOME/.bashrc",
                    "conda activate /science/image/nlp-datasets/creoles/env/creole",
                    "which python",
                    "cd $HOME/creole-dro"]
    with open(full_path, "w") as outfile:
        for line in first_print:
            outfile.write(f"{line}\n")
        outfile.write(script)



def eval_ppl():
    heads = ["mixed"] #, "creoleonly"]
    creoles = ["naija", "singlish", "haitian"]
    encoders = [("bert-base-multilingual-cased", "mbert")]
    experiments = ["pretrained"] #

    i=1
    out_file_dir = "/science/image/nlp-datasets/creoles/results/runppl/pretrained"
    collection_of_scripts = []

    for head in heads:
        if head == "mixed":
            dro = ["one", "random", "language"]
        elif head == "creoleonly":
            dro = ["one", "random", "collect"]

        creole2base = {'naija': 'en', 'singlish': 'en', 'haitian': 'fr'}
        for creole in creoles:
            base = creole2base[creole]
            for enc in encoders:
                for experiment in experiments:
                    if experiment == "baseline":
                        strategy = "one"
                        out_file_name = f"{i}_{head}_{creole}_{experiment}_mbert_{strategy}.sh"
                        script = f"python eval_baseline_ppl.py --file_path=/science/image/nlp-datasets/creoles/data/dev/{creole}/{creole}_and_all.dev.json \
                            --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/{creole}/{creole}-dictionary.txt \
                            --creole={creole} --base_lang={base} --experiment={experiment}  --group_strategy={strategy} --batch_size=1 \
                            --tokenizer={enc[0]} --from_pretrained={enc[0]} --device=cpu \
                             --checkpoint_dir=/science/image/nlp-datasets/creoles/conll/{head}/{experiment}/{enc[1]}/{creole}"
                        writeSH(out_file_name, head, experiment, creole, strategy, script)
                        i+=1
                        collection_of_scripts.append(out_file_name)

                    elif experiment == "dro":
                        for strategy in dro:
                            out_file_name = f"{i}_{head}_{creole}_{experiment}_mbert_{strategy}.sh"
                            script = f"python eval_baseline_ppl.py --file_path=/science/image/nlp-datasets/creoles/data/dev/{creole}/{creole}_and_all.dev.json \
                             --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/{creole}/{creole}-dictionary.txt \
                             --creole={creole} --base_lang={base} --experiment={experiment}  --group_strategy={strategy} --batch_size=1 \
                             --tokenizer={enc[0]} --from_pretrained={enc[0]} --device=cpu \
                             --checkpoint_dir=/science/image/nlp-datasets/creoles/conll/{head}/{experiment}/{enc[1]}/{creole}"
                            writeSH(out_file_name, head, experiment, creole, strategy, script)
                            i+=1
                            collection_of_scripts.append(out_file_name)

                    elif experiment == "pretrained":
                        strategy = "one"
                        out_file_name = f"{i}_{head}_{creole}_{experiment}_mbert_{strategy}.sh"
                        script = f"python eval_baseline_ppl.py --file_path=/science/image/nlp-datasets/creoles/data/dev/{creole}/{creole}_and_all.dev.json \
                        --creole={creole} --base_lang={base} --experiment={experiment}  --group_strategy={strategy} --batch_size=1 \
                        --tokenizer={enc[0]} --from_pretrained={enc[0]} --device=cpu "
                        writeSH(out_file_name, head, experiment, creole, strategy, script)
                        i+=1
                        collection_of_scripts.append(out_file_name)

    with open(os.path.join(out_file_dir, "run.sh"), "w") as outfile:
        for s in collection_of_scripts:
            outfile.write(f"sbatch {s}\n")



def main():
    heads = ["creoleonly"]#["mixed"] #, "creoleonly"]
    creoles = ["haitian"]#["naija", "singlish", "haitian"]
    encoders = [("bert-base-uncased", "bert")] #[("bert-base-uncased", "bert"), ("bert-base-multilingual-cased", "mbert")]
    experiments = ["baseline", "dro"] #, "pretrained"]

    for head in heads:
        if head == "mixed":
            dro = ["one", "random", "language"]
        elif head == "creoleonly":
            dro = ["one", "random", "collect"]

        creole2base = {'naija': 'en', 'singlish': 'en', 'haitian': 'fr'}
        for creole in creoles:
            base = creole2base[creole]
            for enc in encoders:
                for experiment in experiments:
                    if experiment == "baseline":
                        strategy = "one"
                        script = f"python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/dev/{creole}/{creole}_and_all.dev.json \
                        --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/{creole}/{creole}-dictionary.txt \
                        --creole={creole} --base_lang={base} --experiment={experiment}  --group_strategy={strategy} --batch_size=6 \
                        --tokenizer={enc[0]} --from_pretrained={enc[0]} \
                        --checkpoint_dir=/science/image/nlp-datasets/creoles/conll/{head}/{experiment}/{enc[1]}/{creole}"
                        print(script)
                        print(f'echo "##################"')

                    elif experiment == "dro":
                        for strategy in dro:
                            script = f"python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/dev/{creole}/{creole}_and_all.dev.json \
                            --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/{creole}/{creole}-dictionary.txt \
                             --creole={creole} --base_lang={base} --experiment={experiment}  --group_strategy={strategy} --batch_size=6 \
                             --tokenizer={enc[0]} --from_pretrained={enc[0]} \
                             --checkpoint_dir=/science/image/nlp-datasets/creoles/conll/{head}/{experiment}/{enc[1]}/{creole}"

                            print(script)
                            print(f'echo "##################"')

                    # elif experiment == "pretrained":
                    #     strategy = "one"
                    #     script = f"python eval_baseline.py --file_path=/science/image/nlp-datasets/creoles/data/dev/{creole}/{creole}_and_all.dev.json \
                    #     --dictionary_path=/science/image/nlp-datasets/creoles/data/eval/{creole}/{creole}-dictionary.txt \
                    #     --creole={creole} --base_lang={base} --experiment={experiment}  --group_strategy={strategy} --batch_size=6 \
                    #     --tokenizer={enc[0]} --from_pretrained={enc[0]}"
                    #
                    #     print(script)
                    #     print(f'echo "##################"')


eval_ppl()
