import sys
import numpy as np
import argparse
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import svm
from sklearn.externals import joblib
import os


parser = argparse.ArgumentParser(description='Leaves recognition')
parser.add_argument("-i", "--input", metavar="INPUT_FILE", help="Input csv file. Labels placed in the first column.", default=None, action="store", type=str, required=True, dest="inputFile")
parser.add_argument("-s", "--sep", metavar="Column separator", help="CSV column separator. \\t by default. ", default="\t", action="store", type=str, required=False, dest="sep")
args = parser.parse_args()

data = []
labels = []
with open(args.inputFile) as f:
	for line in f:
		data.append(line.strip().split(args.sep)[1:])
		labels.append(line.strip().split(args.sep)[0])
data = np.array(data).astype(np.float64)
labels = np.array(labels)

numberOfFeatures = data.shape[1]
print("Total number of features: %d" % (numberOfFeatures))

bestScore = 0

# wybór najlepszych cech.
for nf in range(1, numberOfFeatures+1):
    data_new = SelectKBest(chi2, k=nf).fit_transform(data, labels)

    #klasyfikatorek
    rfc = RandomForestClassifier()

    #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    #'min_samples_leaf': [1,2,3], 'min_samples_split': [2,3], , 'bootstrap': [True, False], 'warm_start': [True, False], 
    parameters = {'n_estimators': list(range(1, 201, 10)), 'max_depth': list(range(5,21)), 'criterion': ["gini", "entropy"]}
    clf = GridSearchCV(rfc, parameters, cv=10)
    clf.fit(data_new, labels)
    print(clf.best_estimator_)
    print(clf.best_params_)
    print("features used: %d", nf)
    if (clf.best_score_>bestScore):
        bestScore = clf.best_score_
        bestClassifier = clf
        bestData = data_new
    print(clf.best_score_)



where_to_save_classifier = ("/").join(sys.argv[0].strip().split("/")[:-1])
where_to_save_classifier = where_to_save_classifier if len(where_to_save_classifier)>0 else "."
print("Classifier saved in %s/classifier.pkl" %(where_to_save_classifier))
joblib.dump(bestClassifier, "%s/classifier.pkl" % (where_to_save_classifier))

print("\n")
print(bestData[0:20,:])


        
