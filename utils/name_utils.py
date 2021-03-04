# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:35:46 2019coast.coast_params

@author: cneil
"""
import os


def geotifFileName(site_name, date_start, date_end, band_key):

    if band_key  is None:
        file_name = site_name + '_median_' + date_start.replace('-','') +'_'+ date_end.replace('-','') + '.tif'
    else:
        file_name = site_name + '_median_' + date_start.replace('-','') +'_'+ date_end.replace('-','') + '_' + band_key + '.tif'

    return file_name


def pickleDumpName(pickle_type, site_name, sat_name):

    file_name =  site_name + '_' + pickle_type + '_' + sat_name + '.pkl'

    return file_name


def jpegFilePath(settings, jpeg_type, sat_name, date_start, date_end):

    median_dir_path = settings['median_dir_path']
    # create a folder to store the .jpg images showing the detection
    jpeg_file_path = os.path.join(median_dir_path, 'jpg_files', jpeg_type)
    if not os.path.exists(jpeg_file_path):
        os.makedirs(jpeg_file_path)

    file_name = sat_name + '_' + jpeg_type + '_' + date_start.replace('-','') +'_'+ date_end.replace('-','') + '.jpg'

    jpeg_file_path = os.path.join(jpeg_file_path, file_name)

    return jpeg_file_path


def geojsonFilePath(settings, batch=False):

    results_dir_path = settings['results_dir_path']
    sat_name = settings['sat_name']
    site_name = settings['site_name']
    date_start = settings['dates'][0]
    date_end = settings['dates'][1]

    if sat_name == 'S1':
        data_type = settings['polarisation']
    else:
        data_type = sat_name
        pansharpen = settings['pansharpen']
        if pansharpen:
            data_type += '_PS'
    file_name = site_name + '_shoreline_' + data_type + '_' + \
                date_start.replace('-', '') + '_' + date_end.replace('-', '') + '.geojson'

    if batch:
        shoreline_dir_path = os.path.join(results_dir_path, 'batch_shorelines')
    else:
        shoreline_dir_path = os.path.join(results_dir_path, 'shorelines')

    if not os.path.exists(shoreline_dir_path):
        os.makedirs(shoreline_dir_path)

    file_path = os.path.join(shoreline_dir_path, file_name)

    return file_path
