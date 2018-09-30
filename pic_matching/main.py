# -*- coding: gb18030 -*-
# import system packages
import os
import sys
# import argument parser
import argparse
#from keras.preprocessing.image import array_to_img, img_to_array, load_img
# plot function
#import matplotlib.pyplot as plt
import orb_controller
import sift_controller
import time
import cv2
import glob
from utils import *
from multiprocessing.pool import ThreadPool
#import search

##############################################################################
# define function that does linear search of matched images
##############################################################################
#print(os.path.isfile('/home/lym/��/test.jpg'))
def Linear_search(query_path):
    sift = sift_controller.SIFT()
    result = sift.search(query_path)
    del sift
    return (result)

def orb_Linear_search(query_path):
    orb = orb_controller.ORB()
    result = orb.search(query_path)
    del orb
    return (result)
##############################################################################
# define function that does FLANN search of matched images
##############################################################################
def Linear_search_by_FLANN(query_path):
    sift = sift_controller.SIFT()
    result = sift.fast_search(query_path)
    del sift
    return (result)


def orb_Linear_search_by_FLANN(query_path):
    orb = orb_controller.ORB()
    result = orb.orb_fast_search(query_path)
    del orb
    return (result)

##############################################################################
# define function that does in-memory search of matched images
##############################################################################
def Linear_search_prefetching(query_path):
    sift = sift_controller.SIFT()
    result = sift.inmemory_search(query_path)
    del sift
    return (result)
'''
def surf_Linear_search_prefetching(query_path):
    surf = surf_controller.SURF()
    result = surf.inmemory_search(query_path)
    del surf
    return (result)
'''
##############################################################################
# define function that does parallel search of matched images
##############################################################################
def Parallel_search_prefetching(query_path, num_threads):
	input_list = prefetching(query_path)
	pool = ThreadPool(processes=num_threads)
	pool.map(search.multiprocessing_search, input_list)

##############################################################################
# define function that extract features from images and create feature file for
# each image
##############################################################################
def extract_feature_eachfile():
	sift = sift_controller.SIFT()
	sift.dump_eachfile()
	del sift

def orb_extract_feature_eachfile():
    orb = orb_controller.ORB()
    orb.dump_eachfile()
    del orb
##############################################################################
# define function that extract features from images and save feature in one 
# feature file
# each image
##############################################################################
def extract_feature_onedump():
	sift = sift_controller.SIFT()
	sift.dump_onefile()
	del sift
'''
def surf_extract_feature_onedump():
    surf = surf_controller.SURF()
    surf.dump_onefile()
    del surf
'''


##############################################################################
# argument choices
##############################################################################
YESNO = ['yes', 'no']
MODES = ['extractone', 'extractall', 'lsearch', 'fsearch', 'isearch', 'psearch','orb_fsearch','orb_extractone','orb_lsearch']
IMAGE_PATH = './thumb'

##############################################################################
# plot the images
##############################################################################
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

def plot_matched_image(org_img_path, figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows+1)
    # plot orignal one
    img_np = mpimg.imread(org_img_path)
    axeslist.ravel()[0].imshow(img_np, cmap=plt.jet())
    axeslist.ravel()[0].set_title(org_img_path)
    axeslist.ravel()[0].set_axis_off()

    # plot the matched images
    for ind,title in zip(range(len(figures)), figures):
        img_file = os.path.join(IMAGE_PATH, figures[ind])
        img_np = mpimg.imread(img_file)
        axeslist.ravel()[ind+1].imshow(img_np, cmap=plt.jet())
        axeslist.ravel()[ind+1].set_title(title)
        axeslist.ravel()[ind+1].set_axis_off()
    plt.tight_layout() # optional

    plt.show()
    return



##############################################################################
# generate file name of images
##############################################################################

def generate_image_name(feature_pkl_pd):
    matched_pkl_list = feature_pkl_pd[0].tolist() 
    img_name_list = []
    for filename in matched_pkl_list:
        filename_text, filename_ext= os.path.splitext(filename)
        img_name = filename_text + ".jpg"
        img_name_list.append(img_name)
    matched_image_count = len(img_name_list)
    return matched_image_count, img_name_list

##############################################################################
# main
##############################################################################
if __name__ == "__main__":
	
    # specify the arguments parser
    parser = argparse.ArgumentParser()
    # Required arguments: input data file.
    parser.add_argument(
        "--infile",
        required=True,
        help="Input image filename to search matched images or image directory to extract feature",
        metavar="FILE"
    )

    # Mandatory arg to select CNN model
    parser.add_argument(
        "--mode",
        required=False,
        default="extractone",
        choices=MODES,
        help="mode selection"
    )

    # Optional switch to show image
    parser.add_argument(
        "--show",
        required=False,
        default="no",
        choices=YESNO,
        help="show image"
    )

    # parse arguments
    args = parser.parse_args()
    filename = args.infile

    if os.path.isfile(filename) == True:
        single = 1
    elif os.path.isdir(filename) == True:
        single = 0
    else:
        sys.exit("Error: input file or dir does not exist.")

    if args.mode == 'extractone':
        mode = 'extract one'
    elif args.mode == 'extractall':
        mode = 'extract all'
    elif args.mode == 'lsearch':
        mode = 'linear search'
    elif args.mode == 'fsearch':
        mode = 'flann search'
    elif args.mode == 'isearch':
        mode = 'inmemory search'
    elif args.mode == 'psearch':
        mode = 'parallel search'
    elif args.mode == 'orb_extractone':
        mode = 'orb_extract one'
    elif args.mode == 'orb_lsearch':
        mode = 'orb_linear search'
    elif args.mode == 'orb_fsearch':
        mode = 'orb_flann search'

    else:
        sys.exit("Error: unsupported mode.")

    # Optional switch to show image
    if args.show == 'yes':
        show_img = True
    else:
        show_img = False
  
    #if show_img:
        # display a test image
    #    img = load_img(filename)
    #    imgplot = plt.imshow(img)
    #    plt.show()        
    matched_image_count = 0
    # get start time
    t = time.time()
 
    if mode == 'linear search':
        org_img_path = os.path.join(os.getcwd(), filename)
	    # Linear search
        matched_feature_pd = Linear_search(org_img_path)
        matched_image_count, matched_img_name = generate_image_name(matched_feature_pd)
        print ("Match total %d Image: ", (matched_image_count, matched_img_name))
       # if show_img and matched_image_count > 0:
         #   plot_matched_image(org_img_path, matched_img_name, nrows=matched_image_count)

    elif mode == 'orb_linear search':
            org_img_path = os.path.join(os.getcwd(), filename)
            # Linear search
            matched_feature_pd = orb_Linear_search(org_img_path)
            matched_image_count, matched_img_name = generate_image_name(matched_feature_pd)
            print("Match total %d Image: ", (matched_image_count, matched_img_name))
       #     if show_img and matched_image_count > 0:
            #    plot_matched_image(org_img_path, matched_img_name, nrows=matched_image_count)

    elif mode == 'flann search':
        org_img_path = os.path.join(os.getcwd(), filename)
	    # Linear search by FLANN
        matched_feature_pd = Linear_search_by_FLANN(org_img_path)
        matched_image_count, matched_img_name = generate_image_name(matched_feature_pd)
        print ("Match total %d Image: ", (matched_image_count, matched_img_name))
     #   if show_img and matched_image_count > 0:
           # plot_matched_image(org_img_path, matched_img_name, nrows=matched_image_count)

    elif mode == 'orb_flann search':
        org_img_path = os.path.join(os.getcwd(), filename)
        # Linear search by FLANN
        matched_feature_pd = orb_Linear_search_by_FLANN(org_img_path)
        matched_image_count, matched_img_name = generate_image_name(matched_feature_pd)
        print("Match total %d Image: ", (matched_image_count, matched_img_name))
    #   if show_img and matched_image_count > 0:
    # plot_matched_image(org_img_path, matched_img_name, nrows=matched_image_count)

    elif mode == 'inmemory search':
        org_img_path = os.path.join(os.getcwd(), filename)
    	# Linear search + inmemory prefetching
        matched_img_pd = Linear_search_prefetching(org_img_path)
        matched_image_count, matched_img_name = generate_image_name(matched_img_pd)
        print ("Match total %d Image: ", (matched_image_count, matched_img_name))
     #   if show_img and matched_image_count > 0:
         #   plot_matched_image(org_img_path, matched_img[0], nrows=matched_image_count)
        
    elif mode == 'parallel search':
        img_path = os.path.join(os.getcwd(), filename)
 	    # Parallel search + inmemory prefetching
        num_threads = cv2.getNumberOfCPUs()
        Parallel_search_prefetching(img_path, num_threads)

    elif mode == 'extract one':
        # Extract features
        extract_feature_eachfile()
        print("Finish sift feature extraction")

    elif mode == 'orb_extract one':
        orb_extract_feature_eachfile()
        print("Finish orb feature extraction")

    elif mode == 'extract all':
        # Extract features
        extract_feature_onedump()
        print("Finish sift feature extraction")


    else:
        sys.exit("Error: unsupported mode.")
      
    # print total time spent
    print("Perform {} operation in {:.3f} seconds".format(mode, time.time() - t))

