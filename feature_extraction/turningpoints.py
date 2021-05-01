from scipy.io import wavfile
import numpy as np
import pandas as pd
import os


def turningpoints(x):
  N=0
  tnpts=[]
  for i in range(1, len(x)-1):
     if ((x[i-1] < x[i] and x[i+1] < x[i]) or (x[i-1] > x[i] and x[i+1] > x[i])):
       N += 1
       tnpts.append(x[i])
  return tnpts
  
dataset = pd.read_csv('RawData.csv')
df=pd.DataFrame(dataset)
df= df.drop("live", axis='columns')
live=pd.DataFrame(dataset['live'])

tnPointsList=[]

for index, row in df.iterrows():
    turnPoints = turningpoints(row)
    truncated_arr = turnPoints[:200]
    tnPointsList.append(truncated_arr)
df1=pd.DataFrame(tnPointsList)
final=live.join(df1)
final.to_csv("../data/Turningpoints.csv", index=False)

