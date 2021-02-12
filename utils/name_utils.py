# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:35:46 2019coast.coast_params

@author: cneil
"""
import os




def rootDirPathList(dataPartition, country, region):


    root_dir_path = os.path.join(dataPartition, country, region)

    return root_dir_path


def trainingDirPathList():

    root_dir_path_list = rootDirPathList()
    snapped_dir_path_list = []

    for root_dir_path in root_dir_path_list:
        snapped_dir_path_list.append(os.path.join(root_dir_path,
                                                  'snapped_' + griddingParameters()))

    return snapped_dir_path_list


def resultsDirPath():

    year_dir_path = os.path.join(dataPartition, country, region, yearsName())
    if not os.path.isdir(year_dir_path):
        createDirectories([year_dir_path])
        os.mkdir(year_dir_path)
        printSuccess(f'created: {year_dir_path}')

    results_dir_path = os.path.join(year_dir_path, 'results_' + griddingParameters())

    return results_dir_path


def resultFilePath(file_name):

    file_path = os.path.join(resultsDirPath(), file_name + \
                             griddingParameters() + '.tif')

    return file_path


def griddingParameters():

    version = 'r' + str(resampleFactor)

    return version


def gpsFilePath():

    printError('need to revisit')

    gps_dir_path = os.path.join(rootDirPathList(), 'gps')

    if (os.path.isdir(gps_dir_path)):
        file_list = os.listdir(gps_dir_path)

        if len(file_list) == 0:
            printError('no file found in gps directory')

        elif len(file_list) == 1:
            gps_file_path = os.path.join(gps_dir_path, file_list[0])
        else:
            printError('more than 1 file in gsp directory')

        return gps_file_path

    else:
        printError(f'{gps_dir_path} does not exist')


def yearsName():

    for index, year in enumerate(years):

        if index == 0:
            years_name = year
        else:
            years_name += year[2:]

    return years_name


def yearNames():

    year_names = ''
    for index, year in enumerate(years):
        if index == 0:
            year_names += year
        else:
            year_names += ', ' + year

    return  year_names


def shortFileName(long_file_name):

    if len(long_file_name) > 36:
        short_file_name = long_file_name[:32] + '.tif'
    else:
        short_file_name = long_file_name

    return short_file_name
