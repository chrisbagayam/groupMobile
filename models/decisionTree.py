import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

def make(name_dataset):

    dataset = pd.read_csv(name_dataset)

    dataset = dataset.fillna(0)



    X = dataset
    X= X.drop("live", axis='columns')

    y = dataset['live']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


    # scaler = StandardScaler()
    # scaler.fit(X_train)

    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    clf = DecisionTreeClassifier(random_state=1)
    clf = clf.fit(X_train, y_train)

    filename = 'saved_models/decision_tree_'+name_dataset.replace(".csv","").replace("data/", "")+'.sav'
    pickle.dump(clf, open(filename, 'wb'))

    score  = clf.score(X_test, y_test)
    score = score *100
    # print(score)
    return score