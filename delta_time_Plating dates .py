import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict
import glob



#PATH = r"C:\Users\User\Documents\Forecast2023\Data\Dan"
import os
os.chdir(r"C:\Users\User\Documents\Forecast2023\Data2023")
#df_filled = pd.read_csv(r"C:\Users\User\Documents\Forecast2023\Data2023\PD_train2023_dataset_477695_date.csv")

df_filled = pd.read_csv("./PD_Filled_2023.csv")
list_year =["2013","2014","2015","2016","2017","2018","2019","2020", "2021","2022" ]
list_2013 = []
list_2014 = []
list_2015 = []
list_2016 = []
list_2017 = []
list_2018 = []
list_2019 = []
list_2020 = []
list_2021 = []
list_2022 = []


for yr in list_year:
    y=str(yr)
    df_filled['Pd_'+ y] = pd.to_datetime(df_filled['Pd_'+y], format='%d/%m/%Y').dt.date
    #df_filled['Pd_'+ y] = pd.to_datetime(df_filled['Pd_'+y])

for index, row in df_filled.iterrows():
    # m = (row['Pd_2022']).strftime('%D') 
    # print(m)
    for yr in list_year:
        y = str(yr)
        m = int(row['Pd_'+ y].strftime('%m'))
        d = int(row['Pd_'+ y].strftime('%d'))
        delta =  (m-3)*30 +d
        globals()['list_'+ y].append(delta)
for yr in list_year:
    y=str(yr)
    df_filled['Delta_'+ y] = globals()['list_'+ y]
df_filled.to_csv( './PD_train2023_Dataset_delta.csv', index=False)             
