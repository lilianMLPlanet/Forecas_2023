import numpy as np
import pandas as pd
import datetime

df = pd.read_csv(r"C:\Users\User\Documents\Forecast2023\PD_Results_2023_forecast.csv")

def unix_filt(x):
    orig = datetime.datetime.fromtimestamp(1677621600)
    new = orig + datetime.timedelta(days=x)
    return new

def filter_dates(x):
    m = int(x.strftime('%m'))
    d = int(x.strftime('%d'))
    delta =  (m-3)*30 +d
    return 


# df_merge['date'] =  df_merge['predictions'].apply(unix_filt) 

#df['PAcrops_list_2013'] = np.where(((df['is_2013'] == 'NO')), df['type1_2013'], df['PAcrops_list_2013'] )
#df['std'] = df['std'].astype('float32')
df['std'] = df['std'].astype('float32')
df["pred_month"] =df['predictions'] - 31
df["pred_std"] =df['predictions'] - 0.5*df['std']
df["pred_std"] = df['pred_std'].astype(int)
df['dates_min_std '] = df["pred_std"].apply(unix_filt) 
#df['dates_min_month'] = df["pred_month"].apply(unix_filt) 

df['predictions'] = df['predictions'].astype(int)
df['Delta_2021'] = df['Delta_2021'].astype(int)
df['Delta_2022'] = df['Delta_2022'].astype(int)
df['std'] = df['std'].astype('float32')
df['Prediction_date_adj_std'] = pd.to_datetime(df['Prediction_date'],format='%d/%m/%Y')
df['Prediction_date_adj_std'] = np.where((((df['predictions'] >= 61)&(df['predictions'] <= 91))&((df['Delta_2021'] >= 31)&(df['Delta_2021'] <= 60))), 
                                         df['dates_min_std '], df['Prediction_date_adj_std'])
df['Prediction_date_adj_std'] = np.where((((df['predictions'] >= 61)&(df['predictions'] <= 91))&((df['Delta_2022'] >= 31)&(df['Delta_2022'] <= 60))), 
                                         df['dates_min_std '], df['Prediction_date_adj_std'])

# df['Prediction_date_adj_std'] = np.where((((df['predictions'] >= 61)&(df['predictions'] <= 91))&((df['Delta_2021'] >= 31)&(df['Delta_2021'] <= 60))&(df['std']>30)), 
#                                          df['dates_min_month'], df['Prediction_date_adj_std'])
#df['Prediction_date_adj_std'] = pd.to_datetime(df['Prediction_date_adj_std'])

df = df[['commonland', 'state_code', 'county_code',
       'Pd_2016', 'Pd_2017', 'Pd_2018',
       'Pd_2019', 'Pd_2020', 'Pd_2021', 'Pd_2022', 'Delta_2016', 'Delta_2017', 'Delta_2018', 'Delta_2019',
       'Delta_2020', 'Delta_2021', 'Delta_2022', 'predictions',
       'std', 'final_planting_date', 'Delta_Final_date', 
       'dates_min_std ',       
       'Prediction_date',   
       'Prediction_date_adj_std']]
df.to_csv(r"C:\Users\User\Documents\Forecast2023\PD_Results_2023_forecast_adj3.csv")