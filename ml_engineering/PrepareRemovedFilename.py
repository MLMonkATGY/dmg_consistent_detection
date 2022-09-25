import pandas as pd

import glob
import os


def GetAllWrongLabelFilename():
    baseDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/wrong_label"
    remove = "iter4"
    search = f"{baseDir}/**/*.csv"
    allCSvPath = glob.glob(search, recursive=True)
    allValidCsvPath = [x for x in allCSvPath if remove not in x]
    allDf = [pd.read_csv(x) for x in allValidCsvPath]
    allCLFilterFilename = pd.concat(allDf)
    allCLFilterFilename["reject_remarks"] = "noisy_label"
    allCLFilterFilename["rej_filename"] = allCLFilterFilename["dst_filename"]
    return allCLFilterFilename


def GetAllOODFilename():
    baseDir = "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/OOD_reject"
    search = f"{baseDir}/**/*.csv"
    allCSvPath = glob.glob(search, recursive=True)
    allDf = [pd.read_csv(x) for x in allCSvPath]
    allOODRejects = pd.concat(allDf)
    allOODRejects["reject_remarks"] = "OOD"

    return allOODRejects


df1 = GetAllWrongLabelFilename()
df2 = GetAllOODFilename()
allRejectDf = pd.concat([df1, df2])
allRejectDf.to_csv("./all_rejects.csv")
