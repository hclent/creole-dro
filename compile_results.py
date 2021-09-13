import csv
import os


def main():
    final_rows = []
    out_rows = []
    for experiment in ["mixed"]: # or "creoleonly"
        for creole in ["naija", "singlish", "haitian"]:
            path_to_regular_results_dir = f"/science/image/nlp-datasets/creoles/results"
            path_to_ppl_dir = f"/science/image/nlp-datasets/creoles/pplresults"

            regular_files = [f"{experiment}_{creole}.csv", f"results_{creole}.csv"]
            ppl_file = f"{experiment}_final_{creole}.csv"

            model2PPL = {}

            with open(os.path.join(path_to_ppl_dir, ppl_file), "r") as infile:
                pplreader = csv.reader(infile, delimiter=",")
                for row in pplreader:
                    ppl_model_full = row[0]
                    ppl_model_temp = ppl_model_full.split("/")
                    if len(ppl_model_temp) == 1:
                        ppl_model = ppl_model_full
                    else:
                        start = ppl_model_temp.index(experiment)
                        ppl_model = f"/{('/').join(ppl_model_temp[start:])}"
                    ppl = row[2]
                    model2PPL[ppl_model] = ppl

            for regfile in regular_files:
                with open(os.path.join(path_to_regular_results_dir, regfile), "r") as infile:
                    resultsreader = csv.reader(infile, delimiter=",")
                    next(resultsreader, None) #skip header
                    for row in resultsreader:
                        model = row[0]
                        ppl = row[6]
                        if "mbert" in model and ppl == "0.0":
                            #print(f"OLD ROW: {row}")
                            new_row = row
                            new_row[6] = model2PPL[model]
                            out_row = [experiment, creole] + new_row
                            final_rows.append(out_row)
                            #print(f"HMMM ROW: {row}")
                            #print(f"NEW ROW: {out_row}")
                        else:
                            out_row = [experiment, creole] + row
                            final_rows.append(out_row)

        #now make sure that they are unique for this creole
        keep_rows = []
        checked_checkpoints = []
        for fr in final_rows:
            checkpoint = fr[2]
            if checkpoint not in checked_checkpoints:
                keep_rows.append(fr)
                checked_checkpoints.append(checkpoint)

        [out_rows.append(k) for k in keep_rows]

    final_header = ['experiment', 'creole', 'model', 'dev', 'steps', 'p@1','p@5','p@10','PPL','cd-p@1','cd-p@5','cd-p@10']
    path_to_output = "/science/image/nlp-datasets/creoles/results/finalresults"
    out_file = f"{experiment}_final_results.csv"
    with open(os.path.join(path_to_output, out_file), "w") as outfile:
        filewriter = csv.writer(outfile, delimiter=",")
        filewriter.write(final_header)
        for row in out_rows:
            filewriter.write(row)


main()
