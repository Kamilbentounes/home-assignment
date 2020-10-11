#!usr/bin/python
# -*- coding: utf-8 -*-

import json
import argparse
import csv
import glob
import os 

################ Get all files from assignment_imgs folder
files 		= glob.glob("assignment_imgs/*")

def read_Json(jsonFileName):

	"""
	#####################################################################

		This function extracts and returns information contained in a Json File
		mentioned in it argument.

			Input: 
				- jsonFileName: Path to Json file we want to read
			Ouput:
				- data: information extracted from json file saved in the same hierarchy  

	#####################################################################
	"""

	try:
		with open(jsonFileName) as f:
			print("\n\t\t\t***** Reading '{}' file *****".format(jsonFileName))
			data = json.load(f)
	except:
		raise Exception("ERROR ! When reading the following Json file: {}".format(jsonFileName))
	return data

def read_csv(fileNameCsv):

	"""
	#####################################################################

		This function reads row by row the 'label_mapping.csv' with the following delimiter ';',
		checks if the current label's id contains "tomat" both in French and English, then stores
		the label's id and it class following our detection objective ("1" for tomatoes and "0" 
		for others)

			Input: 
				- fileNameCsv: Path to Csv file we want to read
			Ouput:
				- list_id_labels: List of label's Id with its class  

	#####################################################################
	"""

	list_id_labels = dict()

	with open(fileNameCsv, 'r') as csvfile:
		print("\n\t\t\t***** Reading '{}' file ... *****".format(fileNameCsv))
		spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')

		print("\n\t\t\t***** Extracting information from it ... *****")

		for row in spamreader:											# For each line in the csv

			if ("tomat" in row[1].lower() or "tomat" in row[2].lower()) and "sans sauce tomat" not in row[1].lower():	# If 'Tomat' is contained both in "Labeling_named_fr" and "Labeling_named_en" except "Sans sauce tomat" in "Labeling_named_fr" which is a special case 
				list_id_labels[row[0]] = "1"							# Tomatoes class
			else:
				list_id_labels[row[0]] = "0"							# Other class

	return list_id_labels


def compute_center(listCoordinate):

	"""
	#####################################################################

		This function adapts bounding box's coordinates from (x_left_upper, y_left_upper) 
		to (x_center, y_center) 

			Input: 
				- listCoordinate: bounding box's coordinates from 'img_annotations.csv'
			Ouput:
				- x_center, y_center: bounding box's center coordinates  

	#####################################################################
	"""

	return listCoordinate[0], listCoordinate[1], listCoordinate[0] + listCoordinate[2], listCoordinate[1] + listCoordinate[3] 

def start_preprocess(imageFileName, jsonFileName, csvFile, firstPart):

	"""
	#############################################################################
	
		This function start all pre_process.

		VERY IMPORTANT NOTE: Run this script on two parts:

			The first part generate all '.txt' of each image.
			The second part generate 'train.txt' and 'test.txt'

		Because of the update time of the folder which containts all images and all
		'.txt' recently generated with the first part. Even if files_text is put in
		the count_elements_labels() function.

		Input: 
			- imageFileName: 	path to assignment_imgs folder
			- jsonFileName: 	path to img_annotations.json	
			- csvFile:			path to label_mapping.csv
			- firstPart:		1 for runing the first part, else 0

	#############################################################################
	"""

	image_annotations = read_Json(jsonFileName)

	list_id_labels = read_csv(csvFile)	

	if firstPart == 1:													# Generate '.txt' files for each image

		all_data = len(files)
		each = 0
		print("\n\t\t\t***** Generating '.txt' files for each image ... *****")

		for cpt, element in enumerate(files):							# All images

			if (cpt / all_data * 100) >= each:
				print("\n\t\t\t\t{} % ...".format(each))
				each += 10 

			list_annotation = image_annotations[element.split("\\")[-1]]	# Get the image fileName 
			for number, dictionary in enumerate(list_annotation):
				size = len(list_annotation)

				id_image = dictionary["id"]								# Store Image Id

				label = list_id_labels[id_image]						# Get label

				x_left, y_left, x_right, y_right = compute_center(dictionary["box"])	

				if cpt <= (0.95 * all_data):

					if number == 0:
						with open("pytorch_train.txt", "a+") as file:
							file.write(element.split("\\")[-1] + " " + str(x_left) + "," + str(y_left) + "," + str(x_right) + "," + str(y_right) + "," + str(label) + " ")
					elif number == (size-1):
						with open("pytorch_train.txt", "a+") as file:
							file.write(str(x_left) + "," + str(y_left) + "," + str(x_right) + "," + str(y_right) + "," + str(label) + "\n")
					else:
						with open("pytorch_train.txt", "a+") as file:
							file.write(str(x_left) + "," + str(y_left) + "," + str(x_right) + "," + str(y_right) + "," + str(label) + " ")

				elif cpt <= (0.975 * all_data):

					if number == 0:
						with open("pytorch_valid.txt", "a+") as file:
							file.write(element.split("\\")[-1] + " " + str(x_left) + "," + str(y_left) + "," + str(x_right) + "," + str(y_right) + "," + str(label) + " ")
					elif number == (size-1):
						with open("pytorch_valid.txt", "a+") as file:
							file.write(str(x_left) + "," + str(y_left) + "," + str(x_right) + "," + str(y_right) + "," + str(label) + "\n")
					else:
						with open("pytorch_valid.txt", "a+") as file:
							file.write(str(x_left) + "," + str(y_left) + "," + str(x_right) + "," + str(y_right) + "," + str(label) + " ")

				else:

					if number == 0:
						with open("pytorch_test.txt", "a+") as file:
							file.write(element.split("\\")[-1] + " " + str(x_left) + "," + str(y_left) + "," + str(x_right) + "," + str(y_right) + "," + str(label) + " ")
					elif number == (size-1):
						with open("pytorch_test.txt", "a+") as file:
							file.write(str(x_left) + "," + str(y_left) + "," + str(x_right) + "," + str(y_right) + "," + str(label) + "\n")
					else:
						with open("pytorch_test.txt", "a+") as file:
							file.write(str(x_left) + "," + str(y_left) + "," + str(x_right) + "," + str(y_right) + "," + str(label) + " ")


def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--assignment_imgs', '--ip', type = str, default = "assignment_imgs/", help = "The test image path")	
	parser.add_argument('--jsonPath', '--jp', type = str, default = "img_annotations.json", help = "annotation image Json File name")
	parser.add_argument('--csvFile', '--cp', type = str, default = "label_mapping.csv", help = "Labels csv")
	parser.add_argument('--part', '--p', type = int, required = True, help = "Which part you xant to run")

	args = parser.parse_args()

	if (args.part != 0) and (args.part != 1):
		raise Exception('ERROR ! The argument part must be 0 or 1') 

	start_preprocess(args.assignment_imgs, args.jsonPath, args.csvFile, args.part)
	
if __name__ == '__main__':
	main()




