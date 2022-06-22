@author : Romy Beauté
@contact : romybeaute.univ@gmail.com
@date : 20/06/2022

Donne les explications de chaque script, et de l'ordre de run/utilisation
Les chemins sont à changer selon l'utilisateur


Scripts fmriprep : https://github.com/romybeaute/hypnomed/tree/main/code/irmprep_pipeline
Scripts de visualisation : https://github.com/romybeaute/hypnomed/tree/main/code/Visualize



--------------------------------------------------------------------------------------------------------
********** BIDS_conversion.sh **********
--------------------------------------------------------------------------------------------------------

Permet de convertir les fichiers DICOM en fichiers BIDS, selon la bonne organisation
Les fichiers DICOM sont strockés dans DCMROOT='/mnt/data/romy/hypnomed/MRI_raw/DICOM'

==> lance run_heudiconv.sh, qui lui même va appeler heuristic.py


--------------------------------------------------------------------------------------------------------
********** run_heudiconv.sh ********** 
--------------------------------------------------------------------------------------------------------

"heudiconv is a flexible DICOM converter for organizing brain imaging data into structured directory layouts"
Ref : https://heudiconv.readthedocs.io/en/latest/

- data output : OUTDIR='/mnt/data/romy/hypnomed/MRI_raw/BIDS/'
- where to find the heuristic.py file for heudiconv : HEURISTIC_PATH='/mnt/data/romy/hypnomed/MRI_raw/BIDS/code/heuristic.py'
- where the DICOMs are located : DCMROOT='/mnt/data/romy/hypnomed/MRI_raw/DICOM'
- image heudiconv used : IMG="/mnt/data/romy/singularity_images/heudiconv_latest.sif"

==> va appeler heuristic.py pour agencer les DICOM en BIDS et les classer sous fmap, anat & func


--------------------------------------------------------------------------------------------------------
********** heuristic.py ********** 
--------------------------------------------------------------------------------------------------------

Heuristic evaluator for determining which runs belong where allowed template fields - follow python string module
Va agencer les fichier sous :
- anat --> on prend 1x T1 (3DT1) et 1x T2 (3DT2)
- func --> les tâches qui nous intéressent, ici on garde seulement les resting state (RS) --> 3x rs
- fmap --> les fichiers permettant de créer les field maps, ie T2* --> 2x t2star

/!\ pour fmap : il semble il y avoir trop de fichiers, alors on va utiliser un script qui nous permet de visualiser nos données (en images) et de nous 
fournir les infos des fichiers (écrites) pour déterminer lesquels on veut garder
Pour faire ça, on utilise les fichiers du dossier Visualize : 
- visualize.ipynb pour visualiser les images
- fmri_infos.ipynb pour accéder aux infos écrites



--------------------------------------------------------------------------------------------------------
********** BIDS_fieldmap.py ********** 
--------------------------------------------------------------------------------------------------------

Date : 06/08/2022
Permet d'ajouter les champs manquants dans les .json des fichiers fmaps d'après les spécifications BIDS pour qu'ils soient utilisables par fmriprep.

* modify the .json files of fieldmaps files to use them in fmriprep
* see for ref. https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#types-of-fieldmaps
* this script corresponds to the case 1 from the above link

working_path = '/mnt/data/romy/hypnomed/MRI_raw/BIDS'




--------------------------------------------------------------------------------------------------------
********** execute_fmriprep.sh **********
--------------------------------------------------------------------------------------------------------

Nouveau script fmriprep (modif 17/06/2022)
--> le faire toucher avec :
> bash execute_fmriprep.sh
NB : les outputs et les erreurs pour chaque sujet sont stockées dans le dossier log

==> appelle fmriprep_slurm.sh


--------------------------------------------------------------------------------------------------------
********** fmriprep_slurm.sh **********
--------------------------------------------------------------------------------------------------------

Crée un dossier output après avoir lancé execute_fmriprep : 
/mnt/data/romy/hypnomed/MRI_raw/BIDS/derivatives

Quand fmriprep a fini (ou crash), il crée un fichier .html pour chaque sujet







----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Scripts de visualisation : https://github.com/romybeaute/hypnomed/tree/main/code/Visualize

1. Create visualisation folder, where we will store all the necessary infos for each participant
- input : /mnt/data/romy/hypnomed/MRI_raw/DICOM (va regarder les infos de chaque fichier DICOM pour chaque participant)
- output : /mnt/data/romy/hypnomed/MRI_raw/fmri_infos
Ce dont on a besoin, pour chaque participant : 




--------------------------------------------------------------------------------------------------------
********** visualize.ipynb ********** 
--------------------------------------------------------------------------------------------------------


--------------------------------------------------------------------------------------------------------
********** fmri_infos.ipynb ********** 
--------------------------------------------------------------------------------------------------------
Si tout est normal, on devrait avoir : 
- 3 'rs'
- 2 'T1'
- 2 'T2'
- 2 'T2*'

--> si certains fichiers ont beugués, il est possible que nous en ayons plus (eg 4 T1)
solution : regarder dans le csv 'MRI_database_logs.csv' (crée depuis le fichier log de Prisca), et enlever ceux que l'on ne veux pas en se référrant aux ids des participants






