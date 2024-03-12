import os
import csv
import numpy as np
import openneuro as on
import pandas as pd 

ds_num = 'ds004636'
tag_num = '1.0.2'

subj_list = pd.read_csv('subject_list_selfreg.csv')
# only include subjects, sessions, tasks in subject_list
subj = subj_list.subject
ses = subj_list.session
task = subj_list.task
# include functional
func_include = [
	f'{s}/{se}/func/{s}_{se}_{t}*' for s, se, t in zip(subj, ses, task)
]
# include anat
anat_include = [f'{s}/*/anat/*' for s in subj.unique()]
# include fieldmaps
fmap_include = [f'{s}/*/fmap/*' for s in subj.unique()]
include = func_include + anat_include + fmap_include
# pull metadata
include += ['/*.json']

on.download(dataset=ds_num, tag=tag_num, include=include)









    

