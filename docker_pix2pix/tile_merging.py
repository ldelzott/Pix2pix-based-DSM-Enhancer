#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 00:28:36 2022

@author: dellou
"""

import matplotlib.pyplot as plt 
import os
import cv2 as cv 
import numpy as np



## For DSM21 DSM20 NDVI Merge :
    
tile_filenames_dsm20 = os.listdir("/home/dellou/Documents/Master_Thesis/map_data/generated_files/3_DSM20_STEREO_dataset_for_neural_net/removed_empty_images")
tile_filenames_ndvi = os.listdir("/home/dellou/Documents/Master_Thesis/map_data/generated_files/3_NDVI_dataset_for_neural_net/removed_empty_images")


for i, filename in enumerate(tile_filenames_dsm20):

    current_dsm20_tile = cv.imread(f'/home/dellou/Documents/Master_Thesis/map_data/generated_files/3_DSM20_STEREO_dataset_for_neural_net/removed_empty_images/{tile_filenames_dsm20[i]}')
    current_ndvi_tile = cv.imread(f'/home/dellou/Documents/Master_Thesis/map_data/generated_files/3_NDVI_dataset_for_neural_net/removed_empty_images/{tile_filenames_ndvi[i]}')
    if current_dsm20_tile.shape[0] == 256 and current_dsm20_tile.shape[1] == 256:
        merged_image = np.zeros((256, 512, 3), 'uint8')
        merged_image[0:256, 0:256, :] = current_dsm20_tile
        merged_image[0:256, 256:512, :] = current_ndvi_tile
        #cv.imwrite(f'/home/dellou/Documents/Master_Thesis/map_data/generated_files/4_merged_LIDAR_AND_stereo_NO_OVERLAP/{tile_filenames_dsm21[i].split(".")[0]}'+'.jpeg', merged_image)
        cv.imwrite(f'photogram_ndvi_merged/{filename}', merged_image)
        