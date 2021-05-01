import scipy.io
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


dataset = pd.read_csv('RawData.csv')
df=pd.DataFrame(dataset)
live=pd.DataFrame(dataset['live'])
df= df.drop("live", axis='columns')
scaler=StandardScaler()
scaler.fit(df)
scaled_data=scaler.transform(df)
#scaled_data
pca=PCA(n_components=200)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)
df1=pd.DataFrame(x_pca)
final=live.join(df1)
final.to_csv("../data/PCA.csv", index=False)
