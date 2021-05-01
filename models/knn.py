import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

def make(name_dataset):

    dataset = pd.read_csv(name_dataset)

    dataset = dataset.fillna(0)

    X = dataset
    X= X.drop("live", axis='columns')

    y = dataset['live']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


    scaler = StandardScaler()
    scaler.fit(X_train)
    

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)


    total = 0

    y_pred = classifier.predict(X_test)
    y_pred=list(y_pred)
    y_test=list(y_test)


    for i in range(len(y_pred)):   
        if(y_pred[i]==y_test[i]):
            total =total+1
    
    filename = 'saved_models/knn_'+name_dataset.replace(".csv","").replace("data/", "")+'.sav'
    pickle.dump(classifier, open(filename, 'wb'))

    percentage = total/len(y_pred)
    percentage = percentage *100
    score = percentage

    # print(str(percentage)+" %")
    return score
