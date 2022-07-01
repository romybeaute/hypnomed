import pandas as pd

def get_rs_condition(subj,state):
    """
    Retrieve to which condition run_2 & rs_3 belong to (hypnose or meditation)
    """
    df_path = '/home/romy.beaute/projects/hypnomed/data/hypnomed.csv'
    df = pd.read_csv(df_path,sep=';',index_col='sub_id')
    return df.loc[subj][state]