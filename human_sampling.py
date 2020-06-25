import os
import time

players = ['neural']#'random', "greedy", "patientgreedy", 'neural']
runs = [10, 20, 30, 40]
samples = [500]

N = 100
for sample in samples:
    for _id in range(N):
        for observer in range(4):
            for player in players:
                for run in runs:
                    if os.path.exists(f"human_samples/{player}_{sample}_{run}_{observer}_{_id}.json"):
                        print(f"human_samples/{player}_{sample}_{run}_{observer}_{_id}.json exists")
                        continue
                    os.system(f"python human_play.py -r {run} -p {player} -s {sample} -i {_id} -o {observer}")
                    time.sleep(1)
