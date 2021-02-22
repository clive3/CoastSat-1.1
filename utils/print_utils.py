#!/usr/bin/env python

###############################################################################
###############################################################################
###############################################################################
##                                                                           ##
##                         START OF IMPORTS                                  ##
##                                                                           ##
##                                                                           ##
import sys
from math import ceil
import numpy as np
##                                                                           ##
##                                                                           ##
##                         END OF IMPORTS                                    ##
##                                                                           ##
###############################################################################
###############################################################################
##                                                                           ##
##     print to console levels required                                      ##
##     set 0 for no message or 1 otherwise                                   ##
##                                                                           ##
debug    = 0
info     = 0
progress = 1
success  = 1
##                                                                           ##
##                                                                           ##
###############################################################################
###############################################################################
##                                                                           ##
##                         START OF FUNCTIONS                                ##
##                                                                           ##
###############################################################################
###############################################################################
##                                                                           ##
##                                                                           ##


def printDebug(message):

    ## if 'debug' global parameter is set print the message
    if debug == 1: print(f'debug     --  {message}')


def printInfo(message):

    ## if 'debug' global parameter is set print the message
    if info == 1: print(f'info     ##  {message}')


def printProgress(message):

    ## if 'progress' global parameter is set print the message
    if progress == 1: print(f'progress  >>>  {message}')
        
        
def printSuccess(message):

    ## if 'progress' global parameter is set print the message
    # if ( global_params.success == 1 ): print (f'success   \u2713  {message}')
    if success == 1:
        print('*******')
        print(f'success   ~~~  {message}')
        print('*******')


def printWarning(message):
        
    print('*******')
    # print (f'warning   \u26A0  {message}')
    print(f'WARNING   @@@  {message}')
    print('*******')
        
        
def printError(message):
        
    print('*******')
    # print (f'ERROR     \u2620  {message}')
    print(f'ERROR     XXX  {message}')
    print('*******')
    sys.exit()


def printLine(number):
    
    for i in range(number):
        print('*********************************************************************')

##                                                                           ##
##                                                                           ##
###############################################################################
###############################################################################