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

    #OLD KEYS   
    # t1w = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_T1w')
    
    # rest = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-rest_bold')
    # rest_sbref = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-rest_sbref')
    
    # fa = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-fa_bold')
    # fa_sbref = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-fa_sbref')

    # om = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-om_bold')
    # om_sbref = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-om_sbref')

    # fmap_mag =  create_key('sub-{subject}/{session}/fmap/sub-{subject}_{session}_magnitude')
    # fmap_phase = create_key('sub-{subject}/{session}/fmap/sub-{subject}_{session}_phasediff')

    #NEW KEYS

    #anat
    t1w = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_T1w') #T1 (anatomique)
    t2w = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_T2w') #T2 (anatomique)

    #fmap
    t2star = create_key('sub-{subject}/{session}/fmap/sub-{subject}_{session}_T2starw') #T2* (field map)

    #func
    rs_b1 = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-rs_run-1_bold') #resting state block 1
    rs_b2 = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-rs_run-2_bold') #resting state block 2
    rs_b3 = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-rs_run-3_bold') #resting state block 3


    info = {
            t1w: [],
            t2w: [],
            t2star:[],
            rs_b1: [], 
            rs_b2: [], 
            rs_b3: []
            }
    

    for s in seqinfo:
        
        # T1
        if (('21' in s.series_id) | ('3DT1' in s.series_id)):
            info[t1w] = [s.series_id]

        # T2
        if (('15' in s.series_id) | ('3DT2' in s.series_id)):
            info[t2w] = [s.series_id]

        # T2*
        if (('13' in s.series_id) | ('t2star' in s.series_id)):
            info[t2star] = [s.series_id]
            
        # RS blocks
        if (('8' in s.series_id) | ('rs' in s.series_id)):
            info[rs_b1] = [s.series_id] 
        if (('16' in s.series_id) | ('rs' in s.series_id)):
            info[rs_b2] = [s.series_id]
        if (('22' in s.series_id) | ('rs' in s.series_id)):
            info[rs_b3] = [s.series_id] 
            
            
          
    return info
