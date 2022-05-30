#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sebastien
@modified: romy
"""

import os
import os.path
from os import path,makedirs,listdir
from distutils.dir_util import copy_tree


exe_path = os.path.join(*os.getcwd().split("/")[1:5])
print('Running script in following environment : ',exe_path)

try:
    data_path = os.path.join(exe_path,'data')
except:
    data_path = '/mnt/data/romy/hynomed/MRI_raw' #data stored on the cluster intern

print('Using data from ',data_path)

if not path.isdir(path.join(data_path,'DICOM')):
    makedirs(path.join(data_path,'DICOM'))
    print('Created DICOM file in ',data_path)


for sub_id,subject_folder in enumerate([data for data in listdir(data_path) if data.startswith('IRM')],1):
    subject_folder_BIDS = path.join(data_path,'DICOM',f'sub-{sub_id:02}')
    if not path.isdir(subject_folder_BIDS):
        makedirs(subject_folder_BIDS)
    for session_nb,session_folder in enumerate([session for session in (listdir(path.join(data_path,subject_folder))) if session=='scans'],1):
        session_folder_BIDS = path.join(subject_folder_BIDS,f'ses-{session_nb:03}')
        if not path.isdir(session_folder_BIDS):
                makedirs(session_folder_BIDS)
                print('Created BIDS path folder : ',session_folder_BIDS)
        scans_path = path.join(data_path,subject_folder,'scans') #data before being organized in DICOM
        
        for scan_folder in listdir(scans_path):
            if not (scan_folder.startswith('.')):

                if 'Physio' in scan_folder:
                    files_path = 'secondary'
                else:
                    files_path = 'DICOM'

                scan_folder_BIDS = path.join(session_folder_BIDS,scan_folder)
                if not path.isdir(scan_folder_BIDS):
                    makedirs(scan_folder_BIDS)
                    
                copy_tree(path.join(scans_path,scan_folder,files_path),path.join(session_folder_BIDS,scan_folder))

                    

