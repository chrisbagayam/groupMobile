import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

#imports for training and saving
from models import knn
from models import decisionTree as dt
from models import manathanDistance as md
from models import svm
from models import binaryNeuralNetwork as bnn
from feature_extraction import entropy

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

    def predict_knn(self, inputDataset):
        dataset = entropy.inputFeature(inputDataset)

        # dataset = pd.read_csv(inputDataset)
        dataset = dataset.fillna(0)
        X = dataset
        X= X.drop("live", axis='columns')
        y = dataset['live']
        loaded_model = pickle.load(open("saved_models/knn_"+self.name_dataset_without+".sav", 'rb'))
        result_y = loaded_model.predict(X)
        return self.prediction_results(y, result_y)

    
    

    def prediction_results(self, y, y_pred):
        equal = 0
        false_accept = 0
        false_reject = 0
        for i in range(len(y)):
            print(type(y_pred[i]))
            if(y_pred[i]==0 and y_pred[i]!= y[i]):
                false_reject = false_reject + 1
            
            if(y_pred[i]==1 and y_pred[i]!= y[i]):
                false_accept = false_accept + 1

            if(y_pred[i]==y[i]):
                equal = equal+1
        total  = len(y)
        accuracy = equal*100/total
        fa_rate = false_accept*100/total
        fr_rate = false_reject*100/total
        result = {"accuracy":accuracy, "false_accept":fa_rate, "false_reject":fr_rate}
        return result




    def test_manathan_distance(self):
        loaded_model = pickle.load(open("saved_models/manathan_distance_"+self.name_dataset_without+".sav", 'rb'))
        result = loaded_model.score(self.X_test, self.y_test)
        return result*100
    def predict_manathan_distnace(self, inputDataset):
        dataset = entropy.inputFeature(inputDataset)

        # dataset = pd.read_csv(inputDataset)
        dataset = dataset.fillna(0)
        X = dataset
        X= X.drop("live", axis='columns')
        y = dataset['live']
        loaded_model = pickle.load(open("saved_models/manathan_distance_"+self.name_dataset_without+".sav", 'rb'))
        result_y = loaded_model.predict(X)
        return self.prediction_results(y, result_y)
    
    def test_bnn(self):
        loaded_model = pickle.load(open("saved_models/bnn_"+self.name_dataset_without+".sav", 'rb'))
        result = loaded_model.score(self.X_test, self.y_test)
        return result*100
    def predict_bnn(self, inputDataset):
        dataset = entropy.inputFeature(inputDataset)

        # dataset = pd.read_csv(inputDataset)
        dataset = dataset.fillna(0)
        X = dataset
        X= X.drop("live", axis='columns')
        y = dataset['live']
        loaded_model = pickle.load(open("saved_models/bnn_"+self.name_dataset_without+".sav", 'rb'))
        result_y = loaded_model.predict(X)
        return self.prediction_results(y, result_y)
    
    def test_decision_tree(self):
        loaded_model = pickle.load(open("saved_models/decision_tree_"+self.name_dataset_without+".sav", 'rb'))
        result = loaded_model.score(self.X_test, self.y_test)
        return result*100
    
    def predict_decision_tree(self, inputDataset):
        dataset = entropy.inputFeature(inputDataset)

        # dataset = pd.read_csv(inputDataset)
        dataset = dataset.fillna(0)
        X = dataset
        X= X.drop("live", axis='columns')
        y = dataset['live']
        loaded_model = pickle.load(open("saved_models/decision_tree_"+self.name_dataset_without+".sav", 'rb'))
        result_y = loaded_model.predict(X)
        return self.prediction_results(y, result_y)
    
    def test_svn(self):
        loaded_model = pickle.load(open("saved_models/svm_"+self.name_dataset_without+".sav", 'rb'))
        result = loaded_model.score(self.X_test, self.y_test)
        return result*100
    
    def predict_svm(self, inputDataset):
        dataset = entropy.inputFeature(inputDataset)

        # dataset = pd.read_csv(inputDataset)
        dataset = dataset.fillna(0)
        X = dataset
        X= X.drop("live", axis='columns')
        y = dataset['live']
        loaded_model = pickle.load(open("saved_models/svm_"+self.name_dataset_without+".sav", 'rb'))
        result_y = loaded_model.predict(X)
        return self.prediction_results(y, result_y)