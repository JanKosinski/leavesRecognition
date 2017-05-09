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
import sys

class Leaf:
	"""Class representing leaf and its features"""
	def __init__(self, _img, _species):
		self.img = _img.astype(np.uint8)
		self.contour = _img.astype(np.uint8)
		self.img[self.img>0]=1
		self.species = _species

	def extractFeatures(self):
		image, contours, hierarchy = cv2.findContours(self.contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnt = contours[0]
		cv2.drawContours(image, [cnt], 0, (255,255,255), 1)
		self.contourLen = cv2.arcLength(cnt,True)	#obwod konturu

		epsilon1 = 0.1*cv2.arcLength(cnt,True)	#kontury dopasowywane sa na dwoch poziomach szczegolowosci. Postrzepienie liscia jest mierzone poprzez stosunek otrzymanych dlugosci konturow
		epsilon2 = 0.01*cv2.arcLength(cnt,True)
		approx1 = cv2.approxPolyDP(cnt,epsilon1,True)
		approx2 = cv2.approxPolyDP(cnt,epsilon2,True)
		app = cv2.arcLength(approx1,True)
		if (app): #zabezpieczenie przed dzieleniem przez 0
			self.frayingLevel = float(cv2.arcLength(approx2,True))/float(cv2.arcLength(approx1,True))	#im wyzsza wartosc tym bardziej postrzepione krawedzie
		else:
			self.frayingLevel = 0

		self.convexHull_to_contourLen = cv2.arcLength(cv2.convexHull(cnt), True)	#oblicza stosunek dlugosci convexHull do dlugosci rzeczywistego konturu

		#minimalny opisujacy obiekt prostokat (rotacja mozliwa) i jego dlugosc i szerokosc. Stosunek dlugosci do szerokosci prostokata daje informacje o ksztalcie liscia
		self.aToB = float((cv2.minAreaRect(cnt)[1])[0])/float((cv2.minAreaRect(cnt)[1])[1])

		#promien najmniejszego opisujacego okregu moze swiadczyc o wielkosci liscia
		(x,y),radius = cv2.minEnclosingCircle(cnt)
		self.radius = float(radius)

		#stosunek pola opisanego kola do opisanego prostokata swiadczy o proporcjach liscia
		boxArea = ((cv2.minAreaRect(cnt)[1])[0])*((cv2.minAreaRect(cnt)[1])[1])
		circleArea = math.pi * math.pow(radius,2.0)
		self.boxToCircleArea = float(boxArea)/float(circleArea)


		#what about extent, solidity and equivalent diameter, harris corner detection -> number of corners,


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
				dist[distanceFromImageCenter(a.centroid, myImage)] = a.image.astype(np.uint8)
	leaf = sorted(dist.keys())[0]
	leafImg = dist[leaf]
	return leafImg

def toCSV(_list, _filePath):
	s = "\t";
	file = open(_filePath, 'w')
	for i in _list:
		file.write("%s\t%f\t%f\t%f\t%f\t%f\n"%(i.species, i.frayingLevel, i.convexHull_to_contourLen, i.aToB, i.radius, i.boxToCircleArea))
	file.close()

def main(_directory):
	"""Main program functionality"""
	leaves = []
	if (_directory[-1]!='/'):
		_directory = _directory+'/'
	temp = 0
	mainDir = _directory
	directories = os.listdir(mainDir)
	for myDir in directories:
		if not myDir.startswith('.'):
			for d in os.listdir(mainDir+myDir):
				if not d.startswith('.') and "output" not in d:
					leaf = findLeaf(mainDir+myDir+"/"+d)
					leafCopy = leaf
					leafCopy[leafCopy>0]=255
					temp = temp+1
					workingDirectory = mainDir+"output/"+myDir
					if not os.path.exists(workingDirectory):
						os.makedirs(workingDirectory)
					io.imsave(workingDirectory+"/leaf"+str(temp)+".png", leafCopy)
					obj = Leaf(leaf, myDir)
					obj.extractFeatures()
					leaves.append(obj)
	toCSV(leaves, "/Users/jankosinski/Desktop/output.csv")
	#toCSV(leaves, sys.argv[2])


#main(sys.argv[1])
main("/Users/jankosinski/Desktop/Przedmioty/Informatyka Medyczna/leafsnap-subset1/leafsnap-subset1/")


