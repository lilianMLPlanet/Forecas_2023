from os import walk 
import sys
import json
import os
import glob
import pandas as pd
import numpy as np
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
from data_preprocess import get_data
from train_models import train_model
import constants as c



def main(*args):

    years_list = c.YEAR_LIST

    with open('local_paths.txt','r') as f:
        path_str = f.read()
        path = json.loads(path_str)
        
    afs = get_data( path['proag_data_path'], path['proag_geojson'], path['cropscape_data_path']) 
    ds=afs.create_acrege_forecast_csv()
    df_res = afs.mix_cropscape_data_forecast(path['cropscape_file'], ds )
    df_clean = afs.clean_CI_data(df_res)   
    mtx = afs.Crop_Identification_Data(df_clean,years_list )
    model = train_model(path['acreage_dataset_path'], path['ci_dataset_path'], path['planting_dataset_path'])

    for arg in args:

        if arg == 'train_acreage_model':
            d = model.Acreage_Forecast()
            d.to_csv('./results_acreage_model.csv')

        if arg == 'train_ci_model':
            pred_df =model.Crop_Id_Xgboost(path['XGB_modeldata'])
            repo =  model.CI_report(pred_df, years_list)
            repo.to_csv('./results_ci_model.csv')   

        if arg == 'train_planting_model':
            pred_df = model.PD_Xgboost(path['XGB_regresor'])
            repo.to_csv('./results_PD_model.csv') 

if __name__ == '__main__':
    main(*sys.argv[1:])