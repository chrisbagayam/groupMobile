import scipy.io
import pandas as pd
mat = scipy.io.loadmat('../data/Dataset1.mat')
mat2 = scipy.io.loadmat('../data/sampleAttack.mat')

# print(type(mat["Raw_Data"]))

subjects = mat["Raw_Data"].tolist()
subjects2 = mat2["attackVectors"].tolist()

# print(mat2)

df = pd.DataFrame(columns=["live"])
index = 0
a = []
subjCount = 106
trials = 3
freq = 160
# print(mat2)

# print(subjects2[0][0][0][0])

for i in range(subjCount):
    count = 0
    for j in range(trials):
        d = {}
        for k in range(0, 10*freq, 10):
            d[str(count)] = subjects[i][j][k]
            count+=1
        d['live'] = 1
        df = df.append(d,ignore_index=True)
        


for s in range(6):
    for i in range(subjCount):
        count = 0
        for j in range(trials):
            d = {}
            for k in range(0, 10*freq, 10):
                d[str(count)] = subjects2[s][i][j][k]
                count+=1
            d['live'] = 0
            df = df.append(d,ignore_index=True)
            


df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("../data/clean_dataset.csv", index=False)