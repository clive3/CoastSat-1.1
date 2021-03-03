# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:35:46 2019coast.coast_params

@author: cneil
"""


def geotifFileName(site_name, date_start, date_end, band_key):

    if band_key  is None:
        file_name = site_name + '_median_' + date_start.replace('-','') +'_'+ date_end.replace('-','') + '.tif'
    else:
        file_name = site_name + '_median_' + date_start.replace('-','') +'_'+ date_end.replace('-','') + '_' + band_key + '.tif'

    return file_name


def pickleDumpName(pickle_type, site_name, sat_name):

    file_name =  site_name + '_' + pickle_type + '_' + sat_name + '.pkl'

    return file_name


def jpegFileName(jpeg_type, sat_name, date_start, date_end):

    file_name = sat_name + '_' + jpeg_type + date_start.replace('-','') +'_'+ date_end.replace('-','') + '.jpg'

    return file_name


def geojsonFileName(site_name, sat_name, date_start, date_end):

    file_name = site_name + '_shoreline_' + sat_name +'_' +\
                date_start.replace('-','') +'_'+ date_end.replace('-','') + '.geojson'

    return file_name