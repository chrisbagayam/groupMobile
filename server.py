from flask import Flask, render_template, request
import json
from predict_models import All_Models

app = Flask(__name__)

@app.route("/summary/<name_dataset>", methods=['GET', 'POST'])
def getSummary(name_dataset):
    m = All_Models("data/"+name_dataset+".csv")
    dico = {
        "mannathan_distance":m.test_manathan_distance(),
        "svm": m.test_svn(),
        "bnn": m.test_bnn(),
        "knn":m.test_knn(),
        "decision_tree":m.test_decision_tree()
    }

    dico = json.dumps(dico)


    return dico 


@app.route("/predict/<name_dataset>/<input_dataset>", methods=['GET', 'POST'])
def getNoise(name_dataset, input_dataset):
    m = All_Models("data/"+name_dataset+".csv")
    dico_knn = m.predict_knn("data/"+input_dataset+".csv")
    dico_bnn = m.predict_bnn("data/"+input_dataset+".csv")
    dico_manathan = m.predict_manathan_distnace("data/"+input_dataset+".csv")
    dico_svm = m.predict_svm("data/"+input_dataset+".csv")
    dico_decision = m.predict_decision_tree("data/"+input_dataset+".csv")

    dico = {"knn":dico_knn, "bnn": dico_bnn, "manathan":dico_manathan, "svm":dico_svm, "decision_tree": dico_decision}
    dico["input file"]= input_dataset
    dico = json.dumps(dico)
    
    return dico 


if __name__ =="__main__":
    app.run(debug=True)
