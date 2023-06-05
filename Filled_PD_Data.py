import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict
import glob
import os


# #Merging the data from2023 proag
PATH = r"C:\Users\User\Documents\Forecast2023\Data2023\Proag"
path2 = PATH + "\\proag*csv"   
path1 = r"C:\Users\User\Documents\Forecast2023\Data2023\Proag\clu_pa23_summary.csv"
list_path2 = sorted(glob.glob(path2))
df_22 = pd.read_csv(path1)
df_22 = df_22[['commonland']]


# print('df_22', df_22.shape)
# for p in list_path2:
#     year = p.split(".")[0].split("_")[-1]
#     df_temp = pd.read_csv(p)
#     df_temp = df_temp[['commonland','PApd_0']]
#     df_temp.rename(columns={'PApd_0':'Pd_'+ str(year)}, inplace = True)
    
#     print(df_temp.columns)
#     print('df_'+ str(year), df_temp.shape )
    
#     df_merge = df_22.merge(df_temp, how='left', on='commonland')
#     df_22 = df_merge
#     print('df_merge',df_merge.shape)

# #filled the Missing dates 
import os
os.chdir(r"C:\Users\User\Documents\Forecast2023\Data2023")
# PATH = r'./PD_data_raw.csv'
# df_merge3 = pd.read_csv(PATH )
# df_type = pd.read_csv(r"./proag_2023_clus_PWId.csv")
# df_type.rename(columns={'CommonLandUnitId':'commonland'}, inplace = True)
# print(df_type.columns)
# df_type =df_type[['commonland', 'state_code','county_code' ]]
# df_type=df_type.drop_duplicates()
# # print(df_type.shape)
# # df_merge=pd.read_csv( PATH + "\\PD_joinData_365090.csv"  )
# #df_pd_raw = df_type.merge(df_merge, how='left', on='commonland')
# #######2023 changes ######################
# df_pd_raw = df_merge3.merge(df_type, how='left', on='commonland')
# print(df_pd_raw.shape)
# # df= df_pd_raw.dropna(subset= ['Pd_2022'])
# #df = df.fillna(-1)
# df_pd_raw.to_csv('./PD_data_raw_codes')

# df_pd_raw = pd.read_csv(r"C:\Users\User\Documents\Forecast2023\Data2023\PD_data_raw.csv")
# df_codes = pd.read_csv(r"C:\Users\User\Documents\Forecast2023\Data2023\proag_2023_clus_PWId.csv")
# df_codes.rename(columns={'CommonLandUnitId':'commonland'}, inplace = True)
# df_codes = df_codes[['commonland', 'state_code','county_code' ]]

###############2022 filed##################
# df_pd_raw = pd.read_csv(r"C:\Users\User\Documents\Forecast2023\Data2023\PD_data_raw_codes.csv")
# df = df_pd_raw.fillna(-1)

# list_year =["2013", "2014","2015","2016","2017","2018", "2019","2020","2021","2022" ]
# # list_state =[1,4,5 ]
# # list_county = [12,21]
# list_year_re =[]
# list_state_re =[]
# list_county_re =[]
# list_date_re =[]
# for yr in list_year:
#     y=str(yr)
#     df_temp = df[df['Pd_'+y]!=-1]
#     df_temp['Pd_'+y] = pd.to_datetime(df_temp['Pd_'+y]) 
#     m_y= str(df_temp['Pd_'+y].mean()).split(" ")[0]

#     print('year',df_temp.shape)
#     list_state = df_temp['state_code'].unique().tolist()

#     #print(state_list[0],state_list[1], state_list[2])
#     #list_state = [state_list[0],state_list[1], state_list[2]]
#     #list_state = state_list
#     for state in list_state:
#         state = int(state)
#         dt_state = df_temp[df_temp['state_code']==int(state)]
#         m_s= str(dt_state['Pd_'+y].mean()).split(" ")[0]
#         print('state',state, dt_state.shape)
      
#         list_county = dt_state['county_code'].unique().tolist()
         
#         #list_county = [county_list[0],county_list[1], county_list[2]]
#         #list_county = county_list
#         for county in list_county:
#             county = int(county)
#             m_c= str(dt_state['Pd_'+y].mean()).split(" ")[0]
          
#             df['Pd_'+ y] = np.where(((df['Pd_'+ y] == -1) & (df['state_code'] == state) &(df['county_code'] == county)), m_c, df['Pd_'+ y])
#             print(state, county, m_c)
                        
#             list_year_re.append(y)
#             list_state_re.append(state)
#             list_county_re.append(county)
#             list_date_re.append(m_c)

#         df['Pd_'+ y] = np.where(((df['Pd_'+ y] == -1) & (df['state_code'] == state)), m_s, df['Pd_'+ y])    
#         print(state, m_s)
#         list_year_re.append(y)
#         list_state_re.append(state)
#         list_county_re.append("Missing data")
#         list_date_re.append(m_s)

#     df['Pd_'+ y] = np.where(((df['Pd_'+ y] == -1)), m_y, df['Pd_'+ y])  
#     list_year_re.append(y)
#     list_state_re.append("Missing data")
#     list_county_re.append("Missing data")  
#     list_date_re.append(m_y)
#     #df.to_csv(PATH + '\\PD_'+ y +'.csv', index=False)
# #df.to_csv('./PD_Filled_2023.csv', index=False)   
# df_replace = pd.DataFrame({'year':list_year_re,'state':list_state_re, 'county':list_county_re, ''}) 
# df_replace.to_csv(r"C:\Users\User\Documents\Forecast2023\dates_replecement.csv", index=False)
df_replace = pd.read_csv(r"C:\Users\User\Documents\Forecast2023\dates_replecement.csv")
df_replace['date'] = pd.to_datetime(df_replace['date']) 
df_replace2 = df_replace.groupby(['year']).mean()
df_replace2.to_csv(r"C:\Users\User\Documents\Forecast2023\mean_replecement_dates.csv", index=False)
######### join 2022 and 2023 ##################

# df2 = pd.read_csv(r"C:\Users\User\Documents\Forecast2023\Data2023\PD_Filled_raw.csv")
# df_filled_train =pd.concat([df, df2])

# list_year =["2013", "2014","2015","2016","2017","2018", "2019","2020","2021","2022" ]
# # list_state =[1,4,5 ]