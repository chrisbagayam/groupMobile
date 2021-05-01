from scipy.io import wavfile
import numpy as np
import pandas as pd
import pywt



dataset = pd.read_csv('RawData.csv')
df=pd.DataFrame(dataset)
df= df.drop("live", axis='columns')
live=pd.DataFrame(dataset['live'])

ent_list=[]

for index, row in df.iterrows():
    cA,cD = pywt.dwt(row,'bior6.8','per')
    ent_list.append(cA)
df1=pd.DataFrame(ent_list)
final=live.join(df1)
final.to_csv("../data/Biorthogonal.csv", index=False)