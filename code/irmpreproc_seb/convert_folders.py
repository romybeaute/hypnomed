#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 17:57:31 2021

@author: sebastien
"""

from os import path,makedirs,listdir
from distutils.dir_util import copy_tree

data_path = '/mnt/data/sebastien/LONGIMED/RAW_DATA/'


if not path.isdir(path.join(data_path,'DICOM')):
    makedirs(path.join(data_path,'DICOM'))
    
for subject_folder in listdir(data_path):
    if not (subject_folder == 'DICOM'):
        sub_id = int(subject_folder[1:])
        subject_folder_BIDS = path.join(data_path,'DICOM',f'sub-{sub_id:02}')
        if not path.isdir(subject_folder_BIDS):
            makedirs(subject_folder_BIDS)
        for session_nb,session_folder in enumerate(sorted(listdir(path.join(data_path,subject_folder)))):
            session_folder_BIDS = path.join(subject_folder_BIDS,f'ses-{session_nb+1:03}')
            if not path.isdir(session_folder_BIDS):
                makedirs(session_folder_BIDS)
            scans_path = path.join(data_path,subject_folder,session_folder,'scans')
            if len(listdir(scans_path)) != len(listdir(session_folder_BIDS)):
                for scan_folder in listdir(scans_path):
                    if 'Physio' in scan_folder:
                        files_path = 'resources/secondary/files'
                    else:
                        files_path = 'resources/DICOM/files'
                    scan_folder_BIDS = path.join(session_folder_BIDS,scan_folder)
                    if not path.isdir(subject_folder_BIDS):
                        makedirs(scan_folder_BIDS)
                    copy_tree(path.join(scans_path,scan_folder,files_path),path.join(session_folder_BIDS,scan_folder))
                    

