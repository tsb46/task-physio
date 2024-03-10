import os
import csv
import numpy as np
import openneuro as on
import pandas as pd 

ds_num = 'ds004636'
tag_num = '1.0.2'

subj_list = pd.read_csv('subject_list_selfreg.csv')
# only include subjects in subject_list
subj_include = [f'{s}/*' for s in subj_list]
# include json sidecars
subj_include += [f'*.json']

on.download(dataset=ds_num, tag=tag_num, include=subj_include)









    

