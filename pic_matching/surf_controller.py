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

def parse_pkl(dic):
    return dic["id"], dic["des"]


def pickleloader(pklfile):
    try:
        while True:
            yield pkl.load(pklfile)
    except EOFError:
        pass


def parse_glob(path):
    return path.split("/")[2]


def get_top_k_result(match_list=None, k=10):
    result = (sorted(match_list, key=lambda l: l[1], reverse=True))
    return result[:k]


def prefetching(query_path):
    pkl_file = open("surfdump.pkl", "rb")
    surf = SURF()
    query_img = cv2.imread(query_path, 0)
    query_des = surf.extract(query_img)
    input_list = []
    for idx, contents in enumerate(pickleloader(pkl_file)):
        id, indexed_des = parse_pkl(contents)
        if (indexed_des.all()) == None:
            continue
        input_list.append([query_des, id, indexed_des])
    del surf

    pkl_file.close()

    return input_list

'''
surf = cv2.xfeatures2d.SURF_create(400)
# 找到关键点和描述符
key_query, desc_query = surf.detectAndCompute(img, None)
# 把特征点标记到图片上
img = cv2.drawKeypoints(img, key_query, img)

'''

MATCH_SCORE_THRESHOLD = 50
IMAGE_PATH = './thumb'


class SURF():
    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create()
        self.threshold = 0.75
        self.indexedfolder = './surf'
        self.thumbfolder = IMAGE_PATH

    def dump_eachfile(self):
        img_files = os.path.join(self.thumbfolder, "*.jpg")
        for img_path in glob.glob(img_files):
            img_name = img_path.split("/")[2]
            input_img = cv2.imread(img_path, 0)
            kp, des = self.surf.detectAndCompute(input_img, None)
            img_id = img_name.split('.')[0]
            binfile = img_id + '.pkl'
            path = os.path.join(self.indexedfolder, binfile)
            with open(path, 'wb') as dumpfile:
                pickle.dump(des, dumpfile)

    def dump_onefile(self):
        # surf_path = os.path.join(self.indexedfolder, "surfdump.pkl")
        surf_path = "surfdump.pkl"
        img_files = os.path.join(self.thumbfolder, "*.jpg")
        dumpfile = open(surf_path, "wb")
        for img_path in glob.glob(img_files):
            img_name = img_path.split("/")[2]
            input_img = cv2.imread(img_path, 0)
            kp, des = self.surf.detectAndCompute(input_img, None)
            img_id = img_name.split('.')[0]
            contents = {"id": img_id, "des": des}
            pickle.dump(contents, dumpfile)
        dumpfile.close()

    def read(self, featurepath):
        with open(featurepath, "rb") as dump:
            des = pickle.load(dump)
        return des

    def extract(self, img):
        _, des = self.surf.detectAndCompute(img, None)
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
        pkl_file = open("surfdump.pkl", "rb")
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
        match_list = []
        indexed_list = os.listdir(self.indexedfolder)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=10)
        for idx, feature_file in enumerate(indexed_list):
            feature_path = os.path.join(self.indexedfolder, feature_file)
            features = self.read(feature_path)
            # if (features.all()) == None:
            #	continue
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            try:
                matches = flann.knnMatch(query_des, features, k=2)
            except:  # catch *all* exceptions
                print("exception in knnMatch with feature_path ", (feature_path))
                continue
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
