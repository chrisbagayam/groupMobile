import pandas as pd
import numpy as np 
clean_signal = pd.read_csv("RawData.csv")
clean_signal = pd.DataFrame(clean_signal)
live=pd.DataFrame(clean_signal['live'])
clean_signal= clean_signal.drop("live", axis='columns')
mu, sigma = 0, 0.1
shape = clean_signal.shape
# creating noise with the same dimensions as the dataset
noise = np.random.normal(mu, sigma, [shape[0],shape[1]])
# creating a sample noisy dataset
signal = clean_signal + noise
final_signal=live.join(signal)
# saving the noisy dataset in  a CSV file
final_signal.to_csv("../data/gaussian_noise.csv", index=False)





