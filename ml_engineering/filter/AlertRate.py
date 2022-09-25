import pandas as pd

oodCsv = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/OOD_filter_final_0.6.csv"
df = pd.read_csv(oodCsv)
total = len(df)
df = df[df["isOOD"] == True]
oodCount = len(df)
print(f"Alert rate : {oodCount /total}")
