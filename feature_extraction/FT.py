import numpy as np
import pandas as pd

def make():
    data = pd.read_csv('RawData.csv')
    #
    data_frame = pd.DataFrame(data)
    live=pd.DataFrame(data['live'])
    #removing the first column that contains labels
    data_frame= data.drop("live", axis='columns')

    s = data_frame

    fft = np.fft.fft(s)
    #Converting array to dataframe
    fft_done = pd.DataFrame(np.abs(fft))
    #Conveting array to csv file 
    final=live.join(fft_done)
    final.to_csv('../data/FT.csv',index=False)

def inputFeature(name_dataset):
    data = pd.read_csv('../data/'+name_dataset+'.csv')
    #
    data_frame = pd.DataFrame(data)
    live=pd.DataFrame(data['live'])
    #removing the first column that contains labels
    data_frame= data.drop("live", axis='columns')

    s = data_frame

    fft = np.fft.fft(s)
    #Converting array to dataframe
    fft_done = pd.DataFrame(np.abs(fft))
    #Conveting array to csv file 
    final=live.join(fft_done)
    final.to_csv('../data/input_FT.csv',index=False)




