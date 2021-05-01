import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
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


    model = SVC()
    model.fit(X_train, y_train)
    # print(y_train.unique())
    
    filename = 'saved_models/svm_'+name_dataset.replace(".csv","").replace("data/", "")+'.sav'
    pickle.dump(model, open(filename, 'wb'))


    percentage = model.score(X_test, y_test)
    score  = percentage*100
    # print(str(percentage*100) +" % ")

    return score