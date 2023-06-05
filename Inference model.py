import joblib
import pandas as pd

m = pd.read_csv(r"C:\Users\User\Documents\Forecast2023\Data2023\PD_train2023_Dataset_delta.csv")
print(m.shape)
X_eval = m[['Delta_2016', 'Delta_2017', 'Delta_2018',
       'Delta_2019', 'Delta_2020', 'Delta_2021', 'Delta_2022']].to_numpy(dtype='float')
bst = joblib.load('./Models/Planting_date_Forecast_XGBoost_2022_2023-10ts.joblib')
y_pred_eval = bst.predict(X_eval)
m['predictions'] = y_pred_eval.astype(int)
print(m.shape)
m.to_csv(r"C:\Users\User\Documents\Forecast2023\PD_Results_2023.csv", index=False)
m = pd.read_csv(r"C:\Users\User\Documents\Forecast2023\PD_Results_2023.csv")
m['median']=m.filter(like='Delta').apply(lambda x: x.median(), axis=1) 
m['std']=m.filter(like='Delta').apply(lambda x: x.std(), axis=1)   
dates = pd.read_csv(r"C:\Users\User\Documents\Forecast2023\2023_planting_windows.csv")
dates =dates[['state_code','county_code','final_planting_date']]
df_merge = pd.merge(m,dates, left_on=['state_code','county_code'], right_on = ['state_code','county_code'])
df_merge.shape

df_merge['final_planting_date'] = pd.to_datetime(df_merge['final_planting_date'])
def filter_dates(x):
    m = int(x.strftime('%m'))
    d = int(x.strftime('%d'))
    delta =  (m-3)*30 +d
    return delta

df_merge['Delta_Final_date'] = df_merge['final_planting_date'].apply(filter_dates) 

import datetime
def unix_filt(x):
    orig = datetime.datetime.fromtimestamp(1677621600)
    new = orig + datetime.timedelta(days=x)
    return new

df_merge['date'] =  df_merge['predictions'].apply(unix_filt) 

neg_dfmerge = df_merge[df_merge['date-outside']<0] 
neg_dfmerge["prediction_adjst"] = neg_dfmerge["predictions"] - neg_dfmerge["std"]

import datetime
def unix_filt(x):
    orig = datetime.datetime.fromtimestamp(1677621600)
    new = orig + datetime.timedelta(days=x)
    return new

neg_dfmerge['date'] =  neg_dfmerge['prediction_adjst'].apply(unix_filt)
