import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend import Legend

##mean all Target fot Ps mean

n = 50

j = "asr"  # all sources reduction
alldata = pd.read_csv(f"Resume_Solve2_Sources_Run50_CO.csv")
# creating bool series True for NaN values
# bool_series = pd.notnull(alldata["MeanP1"])
# alldata[bool_series]
datamean = alldata[["s%", "MeanP1", "MeanP2", "MeanP3"]].groupby(["s%"]).agg(['mean', 'std']).round(2).reset_index()
print('------------------------------------------------------------')
print('All Target Ps mean to all sources reduction')
print(datamean)
datamean.to_csv(f"Simulation_Ps_Run50_GP_asr_TargetMean_{j}.csv", index=False)

##analysis coefficient of variation

n = 50
j = "asr"
alldata = pd.read_csv(f"Resume_CV2_Sources_Run50_CO.csv")
# creating bool series True for NaN values
# bool_series = pd.notnull(alldata["MeanP1"])
# alldata[bool_series]
Cv_Max = alldata[["propss", "CV_2"]].groupby(["propss"]).agg(['max', 'std'], ascending=True).round(2).reset_index()
print('------------------------------------------------------------')
print('All Target CV max to all sources')
print(Cv_Max)
Cv_Max.to_csv(f"Simulation_CV_Run50_CO_RangeMax_{j}.csv", index=False)


