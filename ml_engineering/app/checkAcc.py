import pandas as pd

predCsv = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/ml_engineering/app/img_price_pred.csv"
df = pd.read_csv(predCsv)
print(f"total : {len(df)}")
df = df[df["gtLabel"] != -1]
df = df.groupby(["CaseID", "viewName"]).head(1)
print(f"total with gt : {len(df)}")

correctDf = df[df["gtLabel"] == df["pred"]]
print(f"Acc :{len(correctDf) / len(df)}")
uniqueCaseId = len(df["CaseID"].unique())
print(f"Unique CaseID :{uniqueCaseId}")
falsePositive = len(df[(df["gtLabel"] == 0) & (df["pred"] == 1)]) / len(
    df[df["gtLabel"] == 0]
)
falseNegative = len(df[(df["gtLabel"] == 1) & (df["pred"] == 0)]) / len(
    df[df["gtLabel"] == 1]
)
truePositive = len(df[(df["gtLabel"] == 1) & (df["pred"] == 1)]) / len(
    df[df["gtLabel"] == 1]
)
trueNegative = len(df[(df["gtLabel"] == 0) & (df["pred"] == 0)]) / len(
    df[df["gtLabel"] == 0]
)
print(f"TP : {truePositive}")
print(f"TN : {trueNegative}")
print(f"FP : {falsePositive}")
print(f"FN : {falseNegative}")
