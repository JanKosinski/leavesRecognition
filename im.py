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
			self.frayingLevel = cv2.arcLength(approx2,True)/cv2.arcLength(approx1,True)	#im wyzsza wartosc tym bardziej postrzepione krawedzie
		else:
			self.frayingLevel = 0

		self.convexHull_to_contourLen = cv2.arcLength(cv2.convexHull(cnt), True)	#oblicza stosunek dlugosci convexHull do dlugosci rzeczywistego konturu

		#minimalny opisujacy obiekt prostokat (rotacja mozliwa) i jego dlugosc i szerokosc. Stosunek dlugosci do szerokosci prostokata daje informacje o ksztalcie liscia
		self.aToB = ((cv2.minAreaRect(cnt)[1])[0])/((cv2.minAreaRect(cnt)[1])[1])

		#promien najmniejszego opisujacego okregu moze swiadczyc o wielkosci liscia
		(x,y),radius = cv2.minEnclosingCircle(cnt)
		self.radius = int(radius)

		#stosunek pola opisanego kola do opisanego prostokata swiadczy o proporcjach liscia
		boxArea = ((cv2.minAreaRect(cnt)[1])[0])*((cv2.minAreaRect(cnt)[1])[1])
		circleArea = math.pi * math.pow(radius,2)
		self.boxToCircleArea = boxArea/circleArea

		#what about extent, solidity and equivalent diameter


def distanceFromImageCenter(_centroid, _image):
	"""Return the distance of object centroid [tuple (row, col)] from the image center"""
	imageCenter = (_image.shape[0]/2, _image.shape[1]/2)
	distance = abs(_centroid[0]-imageCenter[0]) + abs(_centroid[1]-imageCenter[1])
	return distance

def findLeaf(_filePath):
	"""Return leaf extracted from image (file path)"""
	myImage = io.imread(_filePath)
	myImageGrey = color.rgb2gray(myImage)
	mask = (myImageGrey < 0.7)
	labels = measure.label(mask)
	dist = {}
	for a in measure.regionprops(labels):
		dist[distanceFromImageCenter(a.centroid, myImage)] = a.image.astype(np.uint8)
	leaf = sorted(dist.keys())[0]
	leafImg = dist[leaf]
	return skimage.morphology.closing(leafImg, square(3))

def toCSV(_list, _filePath):
	s = "\t";
	file = open(_filePath, 'w')
	for i in _list:
		file.write(i.species+s+i.frayingLevel+s+i.convexHull_to_contourLen+s+i.aToB+s+i.radius+s+i.boxToCircleArea)
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
				leaf = findLeaf(mainDir+myDir+"/"+d)
				leafCopy = leaf
				leafCopy[leafCopy>0]=255
				if (not skimage.exposure.is_low_contrast(leafCopy)):
					temp = temp+1
					workingDirectory = mainDir+"output/"+myDir
					if not os.path.exists(workingDirectory):
						os.makedirs(workingDirectory)
					if(np.count_nonzero(leaf)/leaf.size < 0.7):	#usuwa nieprawidlowe zdjecia skladajace sie w wiekszosci z 1. To prawdopodbnie nie sa liscie
						if ((leafCopy.shape[0]>100 and leafCopy.shape[1]>100) or 'abies' in workingDirectory or 'cedrus' in workingDirectory or 'picea' in workingDirectory): #usuwa szumy i  smieci. Nie dotyczy iglastych
							#io.imsave(workingDirectory+"/leaf"+str(temp)+".png", leafCopy)
							obj = Leaf(leaf, myDir)
							obj.extractFeatures()
							leaves.append(obj)


main("/Users/jankosinski/Desktop/Przedmioty/Informatyka Medyczna/leafsnap-dataset/dataset/images/lab/")


