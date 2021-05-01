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


@app.route("/noise/<name_dataset>", methods=['GET', 'POST'])
def getNoise(name_dataset):
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


if __name__ =="__main__":
    app.run(debug=True)
