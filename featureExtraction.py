import cv2
import numpy as np
import skimage
import skimage.io as io
import skimage.data as data
import skimage.color as color
import skimage.measure as measure
from skimage.morphology import square
from skimage import exposure
import os
from matplotlib import pyplot as plt
import math
import skimage.morphology as morph
import argparse

parser = argparse.ArgumentParser(description='Leaves recognition')
parser.add_argument("-i", "--input", metavar="INPUT_DIR", help="Input directory. Each folder in INPUT_DIR should contain leaves photos of one species and should be properly named (latin species names).", default=None, action="store", type=str, required=True, dest="inputDir")
parser.add_argument("-o", "--output", metavar="OUTPUT_FILE", help="Output file (csv).", default=None, action="store", type=str, required=True, dest="outputFile")
parser.add_argument("-a", "--alternative", metavar="Alternative input directory structure", help="Input directory has alternative structure. Contains only jpg image files. If you are not advanced user please use deafault value.", default=False, action="store", type=bool, required=False, dest="alternative")
args = parser.parse_args()

class Leaf:
	"""Class representing leaf and its features"""
	def __init__(self, _img, _species):
		global testID
		self.img = _img.astype(np.uint8)
		self.contour = _img.astype(np.uint8)
		self.img[self.img>0]=1
		self.species = _species

	def extractFeatures(self, _regionprops):
		image, contours, hierarchy = cv2.findContours(self.contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnt = contours[0]
		cv2.drawContours(image, [cnt], 0, (255,255,255), 1)
		self.contourLen = cv2.arcLength(cnt,True)	#obwod konturu

		epsilon1 = 0.1*cv2.arcLength(cnt,True)	#kontury dopasowywane sa na dwoch poziomach szczegolowosci. Postrzepienie liscia jest mierzone poprzez stosunek otrzymanych dlugosci konturow
		epsilon2 = 0.01*cv2.arcLength(cnt,True)
		approx1 = cv2.approxPolyDP(cnt,epsilon1,True)
		approx2 = cv2.approxPolyDP(cnt,epsilon2,True)
		app = cv2.arcLength(approx1,True)
		self.frayingLevel = 0 if not app else (float(cv2.arcLength(approx2,True))/float(cv2.arcLength(approx1,True)))	#im wyzsza wartosc tym bardziej postrzepione krawedzie

		self.convexHull_to_contourLen = cv2.arcLength(cv2.convexHull(cnt), True)	#oblicza stosunek dlugosci convexHull do dlugosci rzeczywistego konturu

		#minimalny opisujacy obiekt prostokat (rotacja mozliwa) i jego dlugosc i szerokosc. Stosunek dlugosci do szerokosci prostokata daje informacje o ksztalcie liscia
		self.aToB = 0 if ((cv2.minAreaRect(cnt)[1])[1] == 0) else (float((cv2.minAreaRect(cnt)[1])[0])/float((cv2.minAreaRect(cnt)[1])[1]))

		#promien najmniejszego opisujacego okregu moze swiadczyc o wielkosci liscia
		(x,y),radius = cv2.minEnclosingCircle(cnt)
		self.radius = float(radius)

		#stosunek pola opisanego kola do opisanego prostokata swiadczy o proporcjach liscia
		boxArea = ((cv2.minAreaRect(cnt)[1])[0])*((cv2.minAreaRect(cnt)[1])[1])
		circleArea = math.pi * math.pow(radius,2.0)

		self.boxToCircleArea = 0 if (circleArea == 0) else (float(boxArea)/float(circleArea))

		# erozja i zliczanie duzych obiektow po erozji
		erosed = morph.binary_erosion(self.img,np.ones((9,9),np.uint8))
		labels = measure.label(erosed)
		objects = 0
		for a in measure.regionprops(labels):
			if a.area>0.05*self.img.shape[0]*self.img.shape[1]:
				objects = objects + 1
		self.numOfLabelsAfterErosion = 1 if (objects==0) else objects

		#Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
		self.eccentricity = _regionprops.eccentricity

		#major_axis_length. The length of the major axis of the ellipse that has the same normalized second central moments as the region.
		self.major_axis_length = _regionprops.major_axis_length

		#minor_axis_length. The length of the minor axis of the ellipse that has the same normalized second central moments as the region.
		self.minor_axis_length = _regionprops.minor_axis_length

		#extent. Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols)
		self.extent = _regionprops.extent

		#solidity. Ratio of pixels in the region to pixels of the convex hull image.
		self.solidity = _regionprops.solidity

		#filled-area. Number of pixels of filled region.
		self.filled_area = _regionprops.filled_area

		#euler number. Euler characteristic of region. Computed as number of objects (= 1) subtracted by number of holes (8-connectivity).
		self.euler_number = _regionprops.euler_number


def distanceFromImageCenter(_centroid, _image):
	"""Return the distance of object centroid [tuple (row, col)] from the image center"""
	imageCenter = (_image.shape[0]/2, _image.shape[1]/2)
	distance = abs(_centroid[0]-imageCenter[0]) + abs(_centroid[1]-imageCenter[1])
	return distance

def findLeaf(_filePath):
	"""Return leaf extracted from image (file path)"""
	myImage = io.imread(_filePath)
	imgGray = color.rgb2grey(myImage)
	mask = imgGray < skimage.filters.threshold_mean(imgGray)
	imgGray = morph.binary_dilation(mask,np.ones((2,2),np.uint8))
	imgGray = morph.binary_closing(imgGray,np.ones((5,5),np.uint8))
	labels = measure.label(imgGray)
	dist = {}
	areasDictionary = {}
	for b in measure.regionprops(labels):
		areasDictionary[b.filled_area] = b.image
	biggest = sorted(areasDictionary.keys(), reverse=True)[0]
	secBiggest = sorted(areasDictionary.keys(), reverse=True)[1]
	for a in measure.regionprops(labels):
		if (a.bbox[2]<0.9*myImage.shape[0] and a.bbox[3]<0.9*myImage.shape[1]):
			if(a.filled_area == biggest or a.filled_area == secBiggest):
				#dist[distanceFromImageCenter(a.centroid, myImage)] = a.image.astype(np.uint8)
				dist[distanceFromImageCenter(a.centroid, myImage)] = a
	leaf = sorted(dist.keys())[0]
	leafImg = dist[leaf].image.astype(np.uint8)
	return (leafImg, dist[leaf])

def toCSV(_list, _filePath):
	file = open(_filePath, 'w')
	for i in _list:
		file.write("%s\t%f\t%f\t%f\t%f\t%f\t%d\t%f\t%f\t%f\t%f\t%f\t%d\n"%(i.species, i.frayingLevel, i.convexHull_to_contourLen, i.aToB, i.radius, i.boxToCircleArea, i.numOfLabelsAfterErosion, i.eccentricity, i.minor_axis_length, i.major_axis_length, i.extent, i.solidity, i.filled_area))
	file.close()

def main(_directory):
	"""Main program functionality"""
	leaves = []
	if (_directory[-1]!='/'):
		_directory = _directory+'/'
	if (not args.alternative):
		subDirectories = os.listdir(_directory)
		for subDir in subDirectories:
			if not subDir.startswith('.'):
				for photoFile in os.listdir(_directory+subDir):
					if not photoFile.startswith('.'):
						leaf, regionprops = findLeaf(_directory+subDir+"/"+photoFile)
						obj = Leaf(leaf, subDir)
						obj.extractFeatures(regionprops)
						leaves.append(obj)
	else:
		files = os.listdir(_directory)
		for f in files:
			if not f.startswith('.'):
				leaf, regionprops = findLeaf(_directory+f)
				obj = Leaf(leaf, "???")
				obj.extractFeatures(regionprops)
				leaves.append(obj)
	toCSV(leaves, args.outputFile)


main(args.inputDir)


