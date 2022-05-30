#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#test for git
"""
Created on Fri Apr  2 17:57:31 2021
Modified on Fri May 20 

@author: sebastien
@modified: romy
"""
import os
from os import path,makedirs,listdir
from distutils.dir_util import copy_tree


'''
Doc cluster : https://wiki.crnl.fr/lib/exe/fetch.php?media=wiki:informatique:services:doc_cluster.pdf
'''

data_path_cluster = '/crnldata/dycog/meditation/meditation_hypnosis/2016_iEEG_medit_hypno/MRI/MRI_raw/' #accès à la baie CRNL
stockage_interne_cluster = '/mnt/data/romy' #stockage interne au cluster

sessions = False #enter True if different sessions for this paradigm

try:
    data_path = os.path.join(os.getcwd(),'MRI_raw')
except:
    data_path = os.path.join(stockage_interne_cluster,'hynomed/MRI_raw')

print('Data in : ',data_path)


if not path.isdir(path.join(data_path,'DICOM')):
    print('Creating DICOM data path : ',os.path.join(data_path,'DICOM'))
    makedirs(path.join(data_path,'DICOM'))

# irm_data_path = [file for file in listdir(data_path) if file.startswith('IRM')]
    
for sub_id,subject_folder in enumerate([data for data in listdir(data_path) if data.startswith('IRM')],1):
    # print(sub_id,subject_folder)
    # if not (subject_folder == 'DICOM'):
        # sub_id = int(subject_folder[1:])
    subject_id = subject_folder[12:17]
    subject_folder_BIDS = path.join(data_path,'DICOM',f'sub-{sub_id:02}') #Organizing data in DICOM
    
    if not path.isdir(subject_folder_BIDS): 
        makedirs(subject_folder_BIDS)
        print('Created BIDS folder for subject {} : {}'.format(subject_id,subject_folder_BIDS))



    # #for session_nb,session_folder in enumerate(sorted(listdir(path.join(data_path,subject_folder)))):
    # if sessions == True:
    #     for session_nb,session_folder in enumerate([scan for scan in listdir(path.join(data_path,subject_folder)) if scan=='scans']):
    #         session_folder_BIDS = path.join(subject_folder_BIDS,f'ses-{session_nb+1:03}') #Create folder for each session (stored in scans)
    #         if not path.isdir(session_folder_BIDS):
    #             makedirs(session_folder_BIDS)
    #             print('Created BIDS session folder (session {}) for subject {} : {}'.format(session_nb,subject_id,session_folder_BIDS))
    
    #     scans_path = path.join(data_path,subject_folder,session_folder,'scans')

    #     if len(listdir(scans_path)) != len(listdir(session_folder_BIDS)):
    #         for scan_folder in listdir(scans_path):
    #             if 'Physio' in scan_folder:
    #                 files_path = 'resources/secondary/files'
    #             else:
    #                 files_path = 'resources/DICOM/files'
    #             scan_folder_BIDS = path.join(session_folder_BIDS,scan_folder)
    #             if not path.isdir(subject_folder_BIDS):
    #                 makedirs(scan_folder_BIDS)
    #             copy_tree(path.join(scans_path,scan_folder,files_path),path.join(session_folder_BIDS,scan_folder))
                    
    # else: #unique sessions
    scans_path = path.join(data_path,subject_folder,'scans')

    for scan_folder in listdir(scans_path):
        if not (scan_folder.startswith('.')):
            scan_folder_BIDS = path.join(subject_folder_BIDS,scan_folder)
            if not path.isdir(scan_folder_BIDS):
                makedirs(scan_folder_BIDS)
                copy_tree(path.join(scans_path,scan_folder),scan_folder_BIDS)