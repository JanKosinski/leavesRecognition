import argparse
from sklearn.externals import joblib
import os
import numpy as np

parser = argparse.ArgumentParser(description='Leaves classification')
parser.add_argument("-i", "--input", metavar="INPUT_DIRECTORY", help="Input directory containing jpg images to classify", default=None, action="store", type=str, required=True, dest="inputDirectory")
args = parser.parse_args()

scriptDirectory = os.getcwd()
temp_output = "temp_output.csv"
os.system("python3 %s/featureExtraction.py --input %s --output %s/%s --alternative True" % (scriptDirectory.replace(" ", "\ "), args.inputDirectory.replace(" ", "\ "), scriptDirectory.replace(" ", "\ "), temp_output))
try:
	clf = joblib.load("%s/classifier.pkl"%(scriptDirectory))
except IOError:
	print("IOError: You need to create a classifier. Please use scripts featureExtraction.py and leavesRecognition.py to perform this task.\nYou can also create your own classifier and place it in %s"%(scriptDirectory))

examples = []
with open( "%s/%s"%(scriptDirectory, temp_output) ) as f:
    for line in f:
        examples.append(line.strip().split()[1:])
examples = np.asarray(examples)

print(clf.predict(examples))




