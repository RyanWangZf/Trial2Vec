import wget
import os
import pandas as pd

DOC_FILE_URL = 'https://github.com/RyanWangZf/Trial2Vec/raw/main/demo_data/clinical_trial_mini.csv'
LABEL_FILE_URL = 'https://github.com/RyanWangZf/Trial2Vec/raw/main/demo_data/TrialSim-data.xlsx'

def load_demo_data(input_dir='./demo_data'):
    filepath = os.path.join(input_dir, 'clinical_trial_mini.csv')
    label_filepath = os.path.join(input_dir, 'TrialSim-data.xlsx')

    if not os.path.exists(filepath) or not os.path.exists(label_filepath):
        os.makedirs(input_dir)
        filename = wget.download(DOC_FILE_URL, out=filepath)
        print('Download demo data to:', filename)
        filename = wget.download(LABEL_FILE_URL, out=label_filepath)
        print('Download demo data to:', filename)
    
    df = pd.read_csv(filepath, index_col=0)
    df_val = pd.read_excel(label_filepath, index_col=0)
    df_v = pd.DataFrame({'nct_id':df_val.iloc[:,:11].to_numpy().flatten()})
    df_v = df_v.merge(df, on='nct_id', how='inner')
    df_tr = pd.concat([df, df_v], axis=0).drop_duplicates()
    
    return {
        'x': df_tr,
        'fields':['title','intervention_name','disease','keyword'],
        'ctx_fields':['description','criteria'],
        'tag': 'nct_id',
        'x_val':df_val.iloc[:,:11],
        'y_val':df_val.iloc[:,11:],
    }




