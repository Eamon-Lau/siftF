# -*- coding=utf-8 -*-
import numpy as np
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
'''
surf = cv2.xfeatures2d.SURF_create(400)
# 找到关键点和描述符
key_query, desc_query = surf.detectAndCompute(img, None)
# 把特征点标记到图片上
img = cv2.drawKeypoints(img, key_query, img)



detector = cv2.ORB_create()

kp1 = detector.detect(img1, None)
kp2 = detector.detect(img2, None)
kp1, des1 = detector.compute(img1, kp1)
kp2, des2 = detector.compute(img2, kp2)
orb.detectAndCompute
'''


MATCH_SCORE_THRESHOLD = 50
IMAGE_PATH = './thumb'


class ORB():
    def __init__(self):
        self.orb = cv2.ORB_create(2000)
        self.threshold = 0.75
        self.indexedfolder = './orb'
        self.thumbfolder = IMAGE_PATH

    def dump_eachfile(self):
        img_files = os.path.join(self.thumbfolder, "*.jpg")
        for img_path in glob.glob(img_files):
            img_name = img_path.split("/")[2]
            input_img = cv2.imread(img_path, 0)
            kp, desc = self.orb.detectAndCompute(input_img, None)
            try:                                  ####################  I changed
                des = desc.astype(np.float32)
            except:
                print(img_path)
                os.remove(img_path)   ##########################        I changed
            img_id = img_name.split('.')[0]
            binfile = img_id + '.pkl'
            path = os.path.join(self.indexedfolder, binfile)
            with open(path, 'wb') as dumpfile:
                pickle.dump(des, dumpfile)

    def read(self, featurepath):
        with open(featurepath, "rb") as dump:
            des = pickle.load(dump)
        return des

    def extract(self, img):
        #query_img = cv2.imread(img, 0)
        _, desc = self.orb.detectAndCompute(img, None)
        des = desc.astype(np.float32)#################################   I changed
        return des

    def search(self, query_path):
        query_img = cv2.imread(query_path, 0)
        query_des = self.extract(query_img)
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
            for m, n in matches:
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

    def orb_fast_search(self, query_path):
        query_img = cv2.imread(query_path, 0)
        query_des = self.extract(query_img)
        #print ('query_des.shape:',query_des.shape,type(query_des))
        #print ("query_des:",query_des.shape)
        match_list = []
        #print (query_des[1])
        indexed_list = os.listdir(self.indexedfolder)
        for idx , feature_file in enumerate(indexed_list):
            feature_path = os.path.join(self.indexedfolder,feature_file)
            features = self.read(feature_path)


            #print ('features:',features.shape)
            flann = pyflann.FLANN()
            try:
                _, dists = flann.nn(features, query_des, 2, algorithm="kmeans",
                                 branching=5, iterations=1, checks=1);
                #print ('----------------dists---------------------------')
                #print (dists)

            except:
                print("exception in knnMatch with feature_path ", (feature_path))
                continue


            distance_list = dists[:, 0] < self.threshold * dists[:,1]  ########################   i want you advice: if we need threshold?  or just: distance_list = dists[:,0] < 500(or A fixed value.)
            #print (distance_list.shape)
            #print (type(distance_list))
            # print (len(distance_list))
            similar_list = np.sum(distance_list==1)
            match_list.append([feature_file,similar_list])
            del features
        result = get_top_k_result(match_list=match_list,k=3)
        import pandas as pd
        result_pd = pd.DataFrame(result)
        matched = result_pd.loc[result_pd[1] > MATCH_SCORE_THRESHOLD]
        return  matched

#print(ORB().orb_fast_search('/home/lym/下载/siftmatching/18.jpg'),'--------------------------------')
#threshold_np = np(500,[query_des.shape[0],])
#threshold_np1 = np.array([200,200,200,200,200])
