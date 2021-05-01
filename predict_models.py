import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

#imports for training and saving
from models import knn
from models import decisionTree as dt
from models import manathanDistance as md
from models import svm
from models import binaryNeuralNetwork as bnn

class All_Models:
    def __init__(self, name_dataset):
        self.name_dataset = name_dataset
        self.name_dataset_without = self.name_dataset.replace(".csv", "")
        self.name_dataset_without = self.name_dataset_without.replace("data/", "")
        self.X_test = None
        self.y_test = None
        self.set_x_and_y()

        #uncomment to retrain and save the models
        # self.train_and_save_models()
    
    def train_and_save_models(self):
        md.make(self.name_dataset)
        svm.make(self.name_dataset)
        bnn.make(self.name_dataset)
        knn.make(self.name_dataset)
        dt.make(self.name_dataset)

    def set_x_and_y(self):
        dataset = pd.read_csv(self.name_dataset)
        dataset = dataset.fillna(0)
        X = dataset
        X= X.drop("live", axis='columns')
        y = dataset['live']
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.20)

        

    def test_knn(self):
        # load the model from disk
        loaded_model = pickle.load(open("saved_models/knn_"+self.name_dataset_without+".sav", 'rb'))
        result = loaded_model.score(self.X_test, self.y_test)
        return result*100
    
    def test_manathan_distance(self):
        loaded_model = pickle.load(open("saved_models/manathan_distance_"+self.name_dataset_without+".sav", 'rb'))
        result = loaded_model.score(self.X_test, self.y_test)
        return result*100
    
    def test_bnn(self):
        loaded_model = pickle.load(open("saved_models/bnn_"+self.name_dataset_without+".sav", 'rb'))
        result = loaded_model.score(self.X_test, self.y_test)
        return result*100
    
    def test_decision_tree(self):
        loaded_model = pickle.load(open("saved_models/decision_tree_"+self.name_dataset_without+".sav", 'rb'))
        result = loaded_model.score(self.X_test, self.y_test)
        return result*100
    
    def test_svn(self):
        loaded_model = pickle.load(open("saved_models/svm_"+self.name_dataset_without+".sav", 'rb'))
        result = loaded_model.score(self.X_test, self.y_test)
        return result*100