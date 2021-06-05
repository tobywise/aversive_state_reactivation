import pandas as pd
import os
from sklearn.externals import joblib
from pymc3.variational.callbacks import CheckParametersConvergence
import sys
sys.path.insert(0, 'code')
from gp_functions import LatentGPRegression
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_string")
    parser.add_argument("data")
    parser.add_argument('classifier_idx')
    args = parser.parse_args()

    print(os.getcwd())
    print("Loading data")
    df = pd.read_csv(args.data)
    print("Loaded data")
    print(df['classifier_idx'])
    df = df[df['classifier_idx'] == int(args.classifier_idx)]
    print(len(df))
    phase = df['phase'].values[0]
    print(df.columns)
    print(df['shock_received'])
    print("Making data nicer")
    if phase == 'outcome':
        df = df[['arm', 'sequenceness', 'time_point', 'trial_number', 'shock_received', 'chosen', 'Subject', 'abs_pe', 'path_type', 'trial_type']]
        df[['arm', 'sequenceness', 'time_point', 'trial_number', 'shock_received', 'chosen', 'abs_pe', 'path_type', 'trial_type']] = df[['arm', 'sequenceness', 'time_point', 'trial_number', 'shock_received', 'chosen', 'abs_pe', 'path_type', 'trial_type']].astype(float)
        # df['shock_received'] = df['shock_received'].astype(int)
        print(df['shock_received'])
        df.loc[df['shock_received'] == 0, 'shock_received'] = -1
        df.loc[df['chosen'] == 0, 'chosen'] = -1
        df.loc[df['path_type'] == 0, 'path_type'] = -1
        df.loc[df['trial_type'] == 0, 'trial_type'] = -1
        print(df['shock_received'])
    else:
        df = df[['arm', 'sequenceness', 'time_point', 'trial_number', 'chosen', 'Subject', 'path_type', 'trial_type']]

    
    # Center data within subject
    pred_cols = [c for c in df.columns if not c in ['Subject', 'sequenceness', 'time_point']]
    df[pred_cols] = df[pred_cols].astype(float)
        
    # Get observations
    df = pd.merge(df, df.groupby(['trial_number', 'arm']).mean().reset_index().reset_index()[['index', 'trial_number', 'arm']], 
                        on=['trial_number', 'arm']).rename(columns={'index': 'observation_idx'})

    # Decimate

    print("Setting up LGPR")
    lgpr = LatentGPRegression(args.model_string, df, 'time_point', trial_var='observation_idx', scale_x=True)

    print("Fitting LGPR")
    lgpr.fit(n=120000, callbacks=[CheckParametersConvergence()], random_seed=12345)
    trace = lgpr.approx.sample(2000)

    out_name = lgpr.y_var + '__' + '_'.join(lgpr.predictor_vars) + '__' + phase + '__' + args.classifier_idx
    out_name = out_name.replace(':', 'X')
    joblib.dump(trace, 'lgpr_trace___' + out_name)
    joblib.dump(trace, 'lgpr__' + out_name)

    print("DONE")

