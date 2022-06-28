#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:48:54 2021

###########
#permet d'ajouter les champs manquants dans les .json des fichiers fmaps d'après les spécifications BIDS pour qu'ils soient utilisables par fmriprep.
###########

* modify the .json files of fieldmaps files to use them in fmriprep
* see for ref. https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#types-of-fieldmaps
* this script corresponds to the case 1 from the above link
@author: sebastien
@modif: romy
"""
import os 
import json

working_path = '/mnt/data/romy/hypnomed/MRI_raw/BIDS' #path where BIDS data

def get_func_nii(path,ses_folder):
    #return a list with all the functional nifti files
    #it assumes that the fieldmap collected can be used for all of them
    nii_bold_files = []
    for file in os.listdir(path):
        if ('bold' in file) & ('nii.gz' in file):
            
            nii_bold_files.append(os.path.join(ses_folder,'func',file))
    return nii_bold_files

for sub_folder in os.listdir(working_path):
    #loop through subjects and get their fmap and func path; then change .json
    if 'sub-' in sub_folder:
        sub_path = os.path.join(working_path,f'{sub_folder}')
        for ses_folder in os.listdir(sub_path):
            fmap_path = os.path.join(sub_path,ses_folder,'fmap')
            func_path = os.path.join(sub_path,ses_folder,'func')

            for file in os.listdir(fmap_path):
                #magnitude files just need the 'IntendedFor' field
                #phasediff file needs also 'B0FieldIdentifier' field
                if '.json' in file:
                    file_path = os.path.join(fmap_path,file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        data.update({"IntendedFor":get_func_nii(func_path,ses_folder)})
                        if 'phasediff' in file:
                            data.update({"B0FieldIdentifier": "phasediff_fmap0"})
                    with open(file_path, 'w', encoding='utf-8') as json_file:
                        json.dump(data, json_file, ensure_ascii=False, indent=4,sort_keys=True)            
                        