import pandas as pd
import dask.dataframe as dd
from dask.dataframe.core import DataFrame

srcDf: DataFrame = dd.read_csv(
    "/home/alextay96/Desktop/workspace/dmg_price/data/file_metadata.csv",
    low_memory=False,
    # dtype={
    #     "FinalDate": "object",
    #     "DocComment": "object",
    #     "vaAPPEND": "object",
    #     "vaFILENAME": "object",
    #     "vaFILEPATH": "object",
    #     "CaseID": "object",
    #     "Unnamed: 0": "float64",
    #     "iCRTCOID": "object",
    #     "iDOCDEFID": "object",
    #     "iDOCID": "object",
    #     "iFILEID": "object",
    #     "iFILELOCID": "object",
    # },
)
print(type(srcDf))
print(srcDf.columns)
targetFilename = ["Front View", "Rear View"]
srcDf = srcDf[srcDf["StdDocDesc"].isin(targetFilename)]
srcDf.to_csv(
    "/home/alextay96/Desktop/workspace/mrm_workspace/dmg_consistent_detection/data/filesmetadata/selected.csv",
    single_file=True,
)
