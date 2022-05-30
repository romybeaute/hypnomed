import os 

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes


def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """

    t1w = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_T1w')
    
    rest = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-rest_bold')
    rest_sbref = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-rest_sbref')
    
    fa = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-fa_bold')
    fa_sbref = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-fa_sbref')

    om = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-om_bold')
    om_sbref = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-om_sbref')

    fmap_mag =  create_key('sub-{subject}/{session}/fmap/sub-{subject}_{session}_magnitude')
    fmap_phase = create_key('sub-{subject}/{session}/fmap/sub-{subject}_{session}_phasediff')

    
    info = {t1w: [],
            rest: [], fa: [], om: [], 
            rest_sbref: [], fa_sbref: [], om_sbref: [],
            fmap_mag: [], fmap_phase: []
            }
    

    for s in seqinfo:
        
        if (s.dim1 == 320) and ('T1' in s.series_id) and ('NORM' in s.image_type):
            info[t1w] = [s.series_id]
            
            
        if (s.dim4 == 450) and (('normal' in s.series_id) | ('rest' in s.series_id)):
            info[rest] = [s.series_id] 
        if (s.dim4 == 450) and (('ouvert' in s.series_id) | ('om' in s.series_id)):
            info[om] = [s.series_id] 
        if (s.dim4 == 450) and (('compas' in s.series_id) | ('fa' in s.series_id)):
            info[fa] = [s.series_id] 
            
            
        if (s.dim1 == 88) and (s.dim4 == 1) and (('normal' in s.series_id) | ('rest' in s.series_id)):
            info[rest_sbref] = [s.series_id] 
        if (s.dim1 == 88) and (s.dim4 == 1) and (('ouvert' in s.series_id) | ('om' in s.series_id)):
            info[fa_sbref] = [s.series_id] 
        if (s.dim1 == 88) and (s.dim4 == 1) and (('compas' in s.series_id) | ('fa' in s.series_id)):
            info[om_sbref] = [s.series_id] 
            
            
        if (s.dim3 == 104) and (s.dim4 == 1) and ('gre_field_mapping' in s.protocol_name):
            info[fmap_mag] = [s.series_id]
        if (s.dim3 == 52) and (s.dim4 == 1) and ('gre_field_mapping' in s.protocol_name):
            info[fmap_phase] = [s.series_id]
          
          
    return info
