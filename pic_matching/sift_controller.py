# -*- coding=utf-8 -*-
import cv2
import pickle
import os
import glob
from multiprocessing import Pool
import time
from utils import *
# plot function
import matplotlib.pyplot as plt
import pyflann
import numpy as np


MATCH_SCORE_THRESHOLD = 50
IMAGE_PATH = './thumb'

class SIFT():
	def __init__(self):
		self.sift = cv2.xfeatures2d.SIFT_create()
		self.threshold = 0.75
		self.indexedfolder = './sift'
		self.thumbfolder = IMAGE_PATH
	
	def dump_eachfile(self):
		img_files = os.path.join(self.thumbfolder, "*.jpg")
		for img_path in glob.glob(img_files):
			img_name = img_path.split("/")[2]
			input_img = cv2.imread(img_path, 0)
			kp, des = self.sift.detectAndCompute(input_img, None)
			img_id = img_name.split('.')[0]
			binfile = img_id + '.pkl'
			path = os.path.join(self.indexedfolder, binfile) 
			with open(path, 'wb') as dumpfile:
				pickle.dump(des, dumpfile)

	def dump_onefile(self):
		#sift_path = os.path.join(self.indexedfolder, "siftdump.pkl")
		sift_path = "siftdump.pkl"
		img_files = os.path.join(self.thumbfolder, "*.jpg")
		dumpfile = open(sift_path,"wb")
		for img_path in glob.glob(img_files):
			img_name = img_path.split("/")[2]
			input_img = cv2.imread(img_path, 0)
			kp, des = self.sift.detectAndCompute(input_img, None)
			img_id = img_name.split('.')[0]
			contents = {"id" : img_id, "des" : des}
			pickle.dump(contents, dumpfile)
		dumpfile.close()

	def read(self, featurepath):
		with open(featurepath, "rb") as dump:
			des = pickle.load(dump)
		return des
	
	def extract(self, img):
		#query_img = cv2.imread(img, 0)
		_, des = self.sift.detectAndCompute(img, None)
		#print (des.shape)
		#print (np.max(des))

		#print (des)
		#np.set_printoptions(threshold='nan')
		#np.savetxt("filename.txt", des)
		return des
	
	def search(self, query_path):
		query_img = cv2.imread(query_path, 0)
		query_des = self.extract(query_img)
		print (query_des)
		match_list = []
		indexed_list = os.listdir(self.indexedfolder)
		for idx, feature_file in enumerate(indexed_list):
			feature_path = os.path.join(self.indexedfolder, feature_file)
			features = self.read(feature_path)
			if (features.all()) == None:
				continue
			bf = cv2.BFMatcher(cv2.NORM_L2)
			matches = bf.knnMatch(query_des, features, k=2)
			similar_list = []
			for m,n in matches:
				if m.distance < self.threshold * n.distance:
					similar_list.append([m])
			match_list.append([feature_file, len(similar_list)])
			del features, similar_list
		result = get_top_k_result(match_list=match_list, k=5)
		# only return those IDs which matched score is greater than threshold
		import pandas as pd
		result_df = pd.DataFrame(result)
		matched = result_df.loc[result_df[1] > MATCH_SCORE_THRESHOLD]

		return matched
	
	def measure(self, query_des, indexed_list):
		bf = cv2.BFMatcher()
		id = indexed_list[0]
		indexed_des = indexed_list[1]
		matches = bf.knnMatch(query_des, indexed_des, k=2)
		similar_list = []
		for m, n in matches:
			if m.distance < self.threshold * n.distance:
				similar_list.append([m])
		ret = [id, len(similar_list)]
		del indexed_des, similar_list
		return ret
		
	def inmemory_search(self, query_path):
		query_img = cv2.imread(query_path, 0)
		query_des = self.extract(query_img)
		pkl_file = open("siftdump.pkl", "rb")
		indexed_list = []
		for idx, contents in enumerate(pickleloader(pkl_file)):
			id, indexed_des = parse_pkl(contents)
			if (indexed_des.all()) == None:
				continue
			indexed_list.append([id, indexed_des])
		pkl_file.close()
		start_time = time.time()
		match_list = list(map(lambda i: self.measure(query_des, i), indexed_list))
		ret_time = time.time() - start_time
		result = get_top_k_result(match_list=match_list, k=5)
		return result, ret_time

	def fast_search(self, query_path):
		query_img = cv2.imread(query_path, 0)
		query_des = self.extract(query_img)
		# print ('query_des.shape:',query_des.shape,type(query_des))
		# print ("query_des:",query_des.shape)
		match_list = []
		# print (query_des[1])
		indexed_list = os.listdir(self.indexedfolder)
		for idx, feature_file in enumerate(indexed_list):
			feature_path = os.path.join(self.indexedfolder, feature_file)
			features = self.read(feature_path)

			# print ('features:',features.shape)
			flann = pyflann.FLANN()
			try:
				_, dists = flann.nn(features, query_des, 2, algorithm="kmeans",
									branching=5, iterations=1, checks=1);
			# print ('----------------dists---------------------------')
			# print (dists)

			except:
				print("exception in knnMatch with feature_path ", (feature_path))
				continue

			distance_list = dists[:, 0] < self.threshold * dists[:,1]
			# print (type(distance_list))
			# print (len(distance_list))
			similar_list = np.sum(distance_list == 1)
			match_list.append([feature_file, similar_list])
			del features
		result = get_top_k_result(match_list=match_list, k=3)
		import pandas as pd
		result_pd = pd.DataFrame(result)
		matched = result_pd.loc[result_pd[1] > MATCH_SCORE_THRESHOLD]
		return matched



#SIFT().fast_search('/home/lym/下载/siftmatching/17.jpg')
#print(SIFT().fast_search('/home/lym/下载/siftmatching/18.jpg'),'--------------------------------')
SIFT().extract('/home/lym/下载/siftmatching/18.jpg')