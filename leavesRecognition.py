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

# wybór najlepszych cech.
data_new = SelectKBest(chi2, k=5).fit_transform(data, labels)

#Przykladowy podzial na zbiory testowy i uczacy. Do tej pory nieuzywane poniewaz k-krotna walidacja krzyzowa zostala wykorzystana
#data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.25, random_state=1234)

#klasyfikatorek
rfc = RandomForestClassifier()

"""
#badamy wplyw liczby cech na jakosc klasyfikacji
scores = []
n = []
rfc.n_estimators = 50
for i in range(1, data.shape[1]+1):
	data_new = SelectKBest(chi2, k=i).fit_transform(data, labels)
	n.append(i)
	scores.append(cross_val_score(rfc, data_new, labels, cv=10).mean())
	print(n[-1], scores[-1])
plt.plot(n, scores)
plt.title("Wplyw liczby wybranych najlepszych cech na jakosc klasyfikacji")
plt.xlabel("Liczba cech")
plt.ylabel("Jakosc klasyfikacji")
plt.show()"""

#Badanie wpływu parametrów klasyfikatora na wyniki

"""# badamy wplyw liczby drzew na jakosc klasyfikacji
scores = []
n = []
for i in range(1,200,10):
    rfc.n_estimators = i
    n.append(i)
    scores.append(cross_val_score(rfc, data_new, labels, cv=10).mean())
    #print(n[-1], scores[-1])
plt.plot(n, scores)
plt.title("Wplyw liczby drzew na jakosc klasyfikacji")
plt.xlabel("Liczba drzew")
plt.ylabel("Jakosc klasyfikacji")
plt.show()

print("_______________________")
# badamy wplyw liczby drzew na jakosc klasyfikacji
scores = []
n = []
rfc.n_estimators = 50
for i in range(5,21,1):
    rfc.max_depth = i
    n.append(i)
    scores.append(cross_val_score(rfc, data_new, labels, cv=10).mean())
    print(n[-1], scores[-1])
plt.plot(n, scores)
plt.title("Wplyw maksymalnej glebokosci drzew na jakosc klasyfikacji")
plt.xlabel("Maksymalna glebokosc drzew")
plt.ylabel("Jakosc klasyfikacji")
plt.show()"""


#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

#znajdowanie najlepszego klasyfikatora
"""parameters = {'n_estimators': list(range(1, 201, 10)), 'max_depth': list(range(5,21,1)) }
clf = GridSearchCV(rfc, parameters, cv=10)
clf.fit(data_new, labels)
print(clf.best_estimator_)
print(clf.best_params_)
print(clf.best_score_)"""



        
