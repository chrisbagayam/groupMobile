import scipy.io
import numpy as np
import pandas as pd
from scipy.stats import entropy as en
def make():
    dataset = pd.read_csv('RawData.csv')
    df=pd.DataFrame(dataset)
    df= df.drop("live", axis='columns')
    live=pd.DataFrame(dataset['live'])

    ent_list=[]

    for index, row in df.iterrows():
        count=1
        ent_data=[]
        newarr = np.array_split(row,30)
        for i in newarr:
            data=i.value_counts()
            ent=en(data)
            ent_data.append(ent)
        ent_list.append(ent_data)
    df1=pd.DataFrame(ent_list)
    final=live.join(df1)
    final.to_csv("../data/Entropy.csv", index=False)

#input feature extraction
def inputFeature(name_dataset):
    dataset = pd.read_csv(name_dataset)
    df=pd.DataFrame(dataset)
    df= df.drop("live", axis='columns')
    live=pd.DataFrame(dataset['live'])

    ent_list=[]

    for index, row in df.iterrows():
        count=1
        ent_data=[]
        newarr = np.array_split(row,30)
        for i in newarr:
            data=i.value_counts()
            ent=en(data)
            ent_data.append(ent)
        ent_list.append(ent_data)
    df1=pd.DataFrame(ent_list)
    final=live.join(df1)
    return final







if(__name__ == "__main__"):
    make()