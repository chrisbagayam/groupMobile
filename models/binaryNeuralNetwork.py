import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle

def make(name_dataset):
    dataset = pd.read_csv(name_dataset)
    dataset = dataset.fillna(0)

    X = dataset
    X= X.drop("live", axis='columns')
    y = dataset['live']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


    mlp = MLPClassifier(max_iter=500, activation='relu')
    mlp.fit(X_train, y_train)

    filename = 'saved_models/bnn_'+name_dataset.replace(".csv","").replace("data/", "")+'.sav'
    pickle.dump(mlp, open(filename, 'wb'))

    pred = mlp.predict(X_test)
    score = mlp.score(X_test, y_test)
    score = score*100
    # print(str(score)+"%")
    return score