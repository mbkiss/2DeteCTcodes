#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class for reading in the 2DeteCT acquisition settings

Created on Mon Aug 8 11:00:00 2023

@author: mbk
"""

import astra
import numpy as np
import pandas as pd

class SettingsFile_2DeteCT:
    """
    The class creates an SettingsFile object by reading in a .csv file containing the settings
    for one acquisition mode of 2DeteCT and encompasses three main functions:
    1. type_changer is a helper function to return the value of a specific setting parameter in the corresponding type.
    2. get_parameter is a wrapper function that reads in a .csv file and return a specific parameter in its corresponding type.
    3. print_parameter is a function that reads in a .csv file and prints the information stored for a specific parameter.
    """
    def __init__(self, settings_file):
        # Initialization function to read in a .csv file containing the settings
        # for the different acquisition modes of the 2DeteCT data collection.
        self.settings = pd.read_csv(settings_file, sep=';', index_col ='Name', error_bad_lines=False)
    
    @staticmethod
    def type_changer(setting_parameter, type_loc):
        # Function to change the types of a specific value from the settings file
        # into the designated type listed in the settings file stored at type_loc.
        type_map = { 'int': int, 'float': float, 'string': str }
        return type_map[type_loc](setting_parameter)
        
    def get_parameter(self, parameter_name):
        # Wrapper function that uses a settings DataFrame and the parameter_name
        # to return the parameter value in the right type.
        #settings = ReadingSettings_2DeteCT.read_file(settings_file)
        value = self.settings.loc[parameter_name].at["Value"]
        parameter = self.type_changer(value, self.settings.loc[parameter_name].at["Type"])
        return parameter
    
    def print_parameter(self, settings_file, parameter_name):
        #settings = ReadingSettings_2DeteCT.read_file(settings_file)
        parameter = self.settings.loc[parameter_name]
        print(parameter)

