import argparse
from sklearn.externals import joblib
import os
import numpy as np

parser = argparse.ArgumentParser(description='Leaves classification')
parser.add_argument("-i", "--input", metavar="INPUT_DIRECTORY", help="Input directory containing jpg images to classify", default=None, action="store", type=str, required=True, dest="inputDirectory")
parser.add_argument("-d", "--dir", metavar="SCRIPT_DIRECTORY", help="Script directory. Place where featureExtraction.py and leavesRecognition.py are stored.", default=".", action="store", type=str, required=False, dest="scriptDirectory")
args = parser.parse_args()

def main():
	temp_output = "temp_output.csv"
	os.system("python3 %s/featureExtraction.py --input %s --output %s/%s --alternative True" % (args.scriptDirectory.replace(" ", "\ "), args.inputDirectory.replace(" ", "\ "), args.scriptDirectory.replace(" ", "\ "), temp_output))
	try:
		clf = joblib.load("%s/classifier.pkl"%(args.scriptDirectory))
	except IOError:
		print("File %s/classifier.pkl not found" % (args.scriptDirectory))
		print("\nIOError: You need to create a classifier. Please use scripts featureExtraction.py and leavesRecognition.py in order to perform this task.\nYou can also create your own classifier and place it in %s directory\n"%(args.scriptDirectory))
		return
	examples = []
	labels = []
	with open( "%s/%s"%(args.scriptDirectory, temp_output) ) as f:
		for line in f:
			labels.append(line.strip().split()[0])
			examples.append(line.strip().split()[1:])
	examples = np.asarray(examples)

	results = clf.predict(examples)
	for res in range(0, len(results)):
		print(labels[res], results[res])

	os.system("rm %s/%s" % (args.scriptDirectory.replace(" ", "\ "), temp_output))

main()

