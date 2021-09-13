import os
from pathlib import Path
import shutil



def main():
    output_dir = "/science/image/nlp-datasets/creoles/packaged"

    start_path = "/science/image/nlp-datasets/creoles/conll"
    for dir in ["mixed"]:#os.listdir(start_path): #creoleonly , mixed
        for experiment in os.listdir(os.path.join(start_path, dir)): #baseline or dro
            medium_path = os.path.join(start_path, dir)
            
            if experiment == "baseline":
                for enc in ["bert", "mbert"]:
                    for creole in ["naija", "singlish", "haitian"]:
                        cp_checkpoint = os.path.join(start_path, f"{dir}/{experiment}/{enc}/{creole}/100000")
                        cp_location = os.path.join(output_dir, f"{dir}/{experiment}/{enc}/{creole}/100000")
                        # Path(cp_location).mkdir(parents=True, exist_ok=True)
                        destination = shutil.copytree(cp_checkpoint, cp_location)
            
            if experiment == "dro":
                for enc in ["bert", "mbert"]:
                    for creole in ["naija", "singlish", "haitian"]:
                        all_files = os.listdir(os.path.join(start_path, f"{dir}/{experiment}/{enc}/{creole}"))
                        #print(f"ALL: {all_files}")
                        keep_files = [f for f in all_files if "language_100000" in f] #FIXME: collect_100000 for creoleonly
                        print(f"keep_files: {keep_files}")
                        
                        for k in keep_files:
                            base_path = os.path.join(start_path, f"{dir}/{experiment}/{enc}/{creole}")
                            full_path = os.path.join(base_path, k)
                            new_path = os.path.join(output_dir, f"{dir}/{experiment}/{enc}/{creole}")
                            Path(new_path).mkdir(parents=True, exist_ok=True)
                            destination = shutil.copy(full_path, new_path)

main()





