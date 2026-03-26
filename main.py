from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
import os

seed = 0
random.seed(seed)
np.random.seed(seed)

def load_data():
    df = get_input_data()
    return df

def preprocess_data(df):
    df = de_duplication(df)
    df = noise_remover(df)
    return df

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    
    # Text types ensuring
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].fillna('').astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].fillna('').astype('U')
    
    # Store all processed groups to later concatenate and save out.csv
    processed_dfs = []
    
    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:
        print(f"\n{'='*50}\nProcessing dataset: {name}\n{'='*50}")
        X, _ = get_tfidf_features(group_df, fit=True)
        
        # Design Choice 1
        run_chained_multi_outputs(X, group_df, name)
        
        # Design Choice 2 
        run_hierarchical_modeling(X, group_df, name)
            
        processed_dfs.append(group_df)
        
    final_df = pd.concat(processed_dfs)
    final_df.to_csv('out.csv', index=False)
    print("\nSuccessfully finished and saved predictions to out.csv!")

