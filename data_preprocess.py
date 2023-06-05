import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# class get_data with several methods to retrieve and process agricultural data.

class get_data(object):

    def __init__(self, proag_data_path, proag_geojson, cropscape_data_path):
        self.proag_data_path = proag_data_path
        self.proag_geojson = proag_geojson
        self.cropscape_data_path = cropscape_data_path

    #__sum_acreage takes a list of tuples with two elements returns the sum of all the acreages.
    def __sum_acreage(self, x):
        s = 0
        for el in x:
            s+=el[1]
        return s 
    
    # retrive_clu_data takes a string of comma-separated values and converts it into a list.
    
    def retrive_clu_data(self, pa_list_el):
        return ([[x[0].replace("'","").replace("[",""),float(x[1]),x[2],int( x[3].replace("'","").replace("]","") )] for x in [x.split('|') for x in pa_list_el.split(',')]])

    # create_acrege_forecast_csv reads in data from CSV files, calculates acreages, and outputs the result to a CSV file   
    def create_acrege_forecast_csv(self):

        if self.proag_geojson is None:
            return
        sm_2_acr = 0.000247105
        gdf = gpd.read_file(self.proag_geojson)
        #gdf = pd.read_csv(self.proag_geojson)
        gdf = gdf.drop_duplicates()
        gdf =gdf.to_crs(epsg=6933)
        gdf['area'] = gdf['geometry'].apply(lambda x: x.area*sm_2_acr)
        gdf['x_center'] = gdf['geometry'].apply(lambda arg: arg.centroid.x)
        gdf['y_center'] = gdf['geometry'].apply(lambda arg: arg.centroid.y)

       

        path_regrex = self.proag_data_path + "\\proag_glorious_table*csv" # Creates a list of all the files in the alignment folder
        list_path = sorted(glob.glob(path_regrex))

        acreas_dict = defaultdict(list)
        years_acreage_list =[]
        years = []      

        for p in list_path:

            year = p.split(".")[0].split("_")[-1]
            years.append(int(year))
            data = pd.read_csv(p)
            data['acreage'] = (data['pa_list'].apply(lambda x: self.retrive_clu_data(x))).apply(lambda x: self.__sum_acreage(x))
            data['clu_acrege'] = data['acreage'].apply(lambda x: [year,x])
            data.set_index('commonland', inplace=True)
            year_acrage_data = data.to_dict()['clu_acrege']
            years_acreage_list.append(year_acrage_data)
        
        # path2 = self.proag_data_path + "\\proag_comparison*csv"    
        # list_path2 = sorted(glob.glob(path2))
        # for p in list_path2:
                     
        #     year = p.split(".")[0].split("_")[-1]
        #     years.append(int(year))
        #     data = pd.read_csv(p)
        #     data['acreage'] = (data['PA_full_planted_acres_total'])
        #     nan_f = (pd.isna(data['acreage'])) | (  (data['1to1_acres_101_PAplantedTotalAcres']=='NOTOK') &   (data['acreage']==0) )
        #     print(sum(nan_f))
        #     data['acreage'][nan_f] = -1
        #     data['clu_acrege'] = data['acreage'].apply(lambda x: [year, x])
        #     data.set_index('commonland', inplace=True)
        #     year_acrage_data = data.to_dict()['clu_acrege']
        #     years_acreage_list.append(year_acrage_data)

                

        years_acreage_list = [x for _, x in sorted(zip(years, years_acreage_list))]
        for d in years_acreage_list:  # you can list as many input dicts as you want here
            for key, value in d.items():
                acreas_dict[key].append(value)


        years = [str(x) for x in range(2013,2023)]
        is_years = ['is_'+str(x) for x in range(2013, 2023)]
        gdf['num_of_years'] = 0
        for is_year,year in zip(is_years,years):
            gdf[year] = ''
            gdf[is_year] = 'NO'

        diff_acr = []
        for k_r,(_,r) in enumerate(gdf.iterrows()):
            print(k_r)
            if r['commonland'] in acreas_dict:

                acres = acreas_dict[r['commonland']]
                area = r['area']
                gdf['num_of_years'][k_r] = len(acres)
                for year in acres:
                    print(year)
                    if year[1] > area:
                        plant_area  = area
                    elif year[1] < 0:
                        plant_area = 0
                    else:
                        plant_area = year[1]
                    gdf[year[0]][k_r] = plant_area
                    if year[1]==-1:
                        gdf['is_' + year[0]][k_r] = 'NO'
                    else:
                        gdf['is_' + year[0]][k_r] = 'YES'


                  
        
        gdf[['commonland','area','x_center','y_center','stateabbre','countyname','num_of_years']+[str(x) for x in list(range(2013,2023))]+ ['is_'+str(x) for x in range(2013, 2023)]].to_csv(self.proag_geojson.replace('.geojson', '_acreage_data.csv'),index=False)

        save_path = r"C:\Users\User\Documents\Forecast2023\Data2023\Final_Acreage_dataset_2023.csv"
        gdf.to_csv(save_path)
        return gdf
    
    # cropscape_file and a df_proa dataframe as inputs, and it returns a merged dataframe that combines Proag data and cropscape data.
    def mix_cropscape_data_forecast(self, cropscape_file, df_proa ):

        # df_cropscape = pd.read_csv(os.path.join(self.cropscape_data_path,cropscape_file))
        df_proa = df_proa .drop(['Unnamed: 0','geometry','type1_acres_2013','type1_acres_2014','type1_acres_2015' , 'type1_acres_2016','type1_acres_2017','type1_acres_2018',
                                 'type1_acres_2019','type1_acres_2020' , 'type1_acres_2021','type1_acres_2022','geometry', 'area',
       'x_center', 'y_center'],axis=1)
        df_proa = df_proa.rename(columns={'CommonLandUnitId': 'commonland'})
        print(df_proa.columns)
        # df2 = df_proa[['commonland','area','x_center','y_center','stateabbre','countyname','num_of_years','2013','2014',
        #                '2015','2016','2017','2018','2019','2020','2021','2022','is_2013','is_2014','is_2015','is_2016',
        #                'is_2017','is_2018','is_2019','is_2020','is_2021','is_2022']]
        # df_res= df2.merge(df, on='commonland', how='left')
        
        path_regrex = self.proag_data_path + "\\proag*csv" # Creates a list of all the files in the alignment folder
        list_path = sorted(glob.glob(path_regrex))
        dframe = df_proa
        for p in list_path:
            dp = pd.read_csv(p)
            year = p.split('.')[0].split("_")[-1]
            # if year != '2022' :
            #     dp = dp.rename(columns={"proag_commonlandunitid": "commonland", 'pa_list_len':'pa_list_full_len'})
        
            dp1 =dp[[ 'commonland', 'PAcrops_list']]
        
            dp1 = dp1.rename(columns={ "PAcrops_list": "PAcrops_list"+ "_"+year })
      
            print(dp1.columns) 
                         
            df_merge = pd.merge(dframe, dp1, on=['commonland'],  how='left')
            dframe = df_merge       


        return df_merge
    
    # The clean_CI_data function takes a  DataFrame  modifies and replaces values  Nan from Proag qith values extracted from Cropsacape.
    def clean_CI_data(self, df):

        df['type1_2013'] = np.where(((df['type1_2013'] == 'Grassland/Pasture')|(df['type1_2013']== 'Other Hay/Non Alfalfa')), df['type2_2013'], df['type1_2013'])
        df['type1_2014'] = np.where(((df['type1_2014'] == 'Grassland/Pasture')|(df['type1_2014']== 'Other Hay/Non Alfalfa')), df['type2_2014'], df['type1_2014'])
        df['type1_2015'] = np.where(((df['type1_2015'] == 'Grassland/Pasture')|(df['type1_2015']== 'Other Hay/Non Alfalfa')), df['type2_2015'], df['type1_2015'])
        df['type1_2016'] = np.where(((df['type1_2016'] == 'Grassland/Pasture')|(df['type1_2016']== 'Other Hay/Non Alfalfa')), df['type2_2016'], df['type1_2016'])
        df['type1_2017'] = np.where(((df['type1_2017'] == 'Grassland/Pasture')|(df['type1_2017']== 'Other Hay/Non Alfalfa')), df['type2_2017'], df['type1_2017'])
        df['type1_2018'] = np.where(((df['type1_2018'] == 'Grassland/Pasture')|(df['type1_2018']== 'Other Hay/Non Alfalfa')), df['type2_2018'], df['type1_2018'])
        df['type1_2019'] = np.where(((df['type1_2019'] == 'Grassland/Pasture')|(df['type1_2019']== 'Other Hay/Non Alfalfa')), df['type2_2019'], df['type1_2019'])
        df['type1_2020'] = np.where(((df['type1_2020'] == 'Grassland/Pasture')|(df['type1_2020']== 'Other Hay/Non Alfalfa')), df['type2_2020'], df['type1_2020'])
        df['type1_2021'] = np.where(((df['type1_2021'] == 'Grassland/Pasture')|(df['type1_2021']== 'Other Hay/Non Alfalfa')), df['type2_2021'], df['type1_2021'])
        df['type1_2022'] = np.where(((df['type1_2022'] == 'Grassland/Pasture')|(df['type1_2022']== 'Other Hay/Non Alfalfa')), df['type2_2022'], df['type1_2022'])

        df['PAcrops_list_2013'] = np.where(((df['is_2013'] == 'NO')), df['type1_2013'], df['PAcrops_list_2013'] )
        df['PAcrops_list_2014'] = np.where(((df['is_2014'] == 'NO')), df['type1_2014'], df['PAcrops_list_2014'])
        df['PAcrops_list_2015'] = np.where(((df['is_2015'] == 'NO')), df['type1_2015'], df['PAcrops_list_2015'])
        df['PAcrops_list_2016'] = np.where(((df['is_2016'] == 'NO')), df['type1_2016'], df['PAcrops_list_2016'])
        df['PAcrops_list_2017'] = np.where(((df['is_2017'] == 'NO')), df['type1_2017'], df['PAcrops_list_2017'])
        df['PAcrops_list_2018'] = np.where(((df['is_2018'] == 'NO')), df['type1_2018'], df['PAcrops_list_2018'])
        df['PAcrops_list_2019'] = np.where(((df['is_2019'] == 'NO')), df['type1_2019'], df['PAcrops_list_2019'])
        df['PAcrops_list_2020'] = np.where(((df['is_2020'] == 'NO')), df['type1_2020'], df['PAcrops_list_2020'])
        df['PAcrops_list_2021'] = np.where(((df['is_2021'] == 'NO')), df['type1_2021'], df['PAcrops_list_2021'])
        df['PAcrops_list_2022'] = np.where(((df['is_2022'] == 'NO')), df['type1_2022'], df['PAcrops_list_2022'])
        df = df.fillna('0')

        def filter(x):
            y=  str(x).replace("'","").replace("[","").replace("]","").split(",")[0]
            
            return y
  
        df['PAcrops_list_2013'] = df['PAcrops_list_2013'].apply(filter) 
        df['PAcrops_list_2014'] = df['PAcrops_list_2014'].apply(filter) 
        df['PAcrops_list_2015'] = df['PAcrops_list_2015'].apply(filter)
        df['PAcrops_list_2016'] = df['PAcrops_list_2016'].apply(filter) 
        df['PAcrops_list_2017'] = df['PAcrops_list_2017'].apply(filter) 
        df['PAcrops_list_2018'] = df['PAcrops_list_2018'].apply(filter) 
        df['PAcrops_list_2019'] = df['PAcrops_list_2019'].apply(filter) 
        df['PAcrops_list_2020'] = df['PAcrops_list_2020'].apply(filter) 
        df['PAcrops_list_2021'] = df['PAcrops_list_2021'].apply(filter) 
        df['PAcrops_list_2022'] = df['PAcrops_list_2022'].apply(filter)

        return df
    
    # Crop_Identification_Data covert common land, county code, state code, and the numerical data Input of Model
    def Crop_Identification_Data(self, df, yeas_list ):
        #temporal#

        df['PAcrops_list_2013'] = np.where(((df['PAcrops_list_2013'] == '0')), df['type1_2013'], df['PAcrops_list_2013'] )
        df['PAcrops_list_2014'] = np.where(((df['PAcrops_list_2014'] == '0')), df['type1_2014'], df['PAcrops_list_2014'])
        df['PAcrops_list_2015'] = np.where(((df['PAcrops_list_2015'] == '0')), df['type1_2015'], df['PAcrops_list_2015'])
        df['PAcrops_list_2016'] = np.where(((df['PAcrops_list_2017'] == '0')), df['type1_2017'], df['PAcrops_list_2017'])
        df['PAcrops_list_2018'] = np.where(((df['PAcrops_list_2018'] == '0')), df['type1_2018'], df['PAcrops_list_2018'])
        df['PAcrops_list_2019'] = np.where(((df['PAcrops_list_2019'] == '0')), df['type1_2019'], df['PAcrops_list_2019'])
        df['PAcrops_list_2020'] = np.where(((df['PAcrops_list_2020'] == '0')), df['type1_2020'], df['PAcrops_list_2020'])
        df['PAcrops_list_2021'] = np.where(((df['PAcrops_list_2021'] == '0')), df['type1_2021'], df['PAcrops_list_2021'])
        df['PAcrops_list_2022'] = np.where(((df['PAcrops_list_2022'] == '0')), df['type1_2022'], df['PAcrops_list_2022'])

        df['PAcrops_list_2013'] = np.where(((df['PAcrops_list_2013'] == 0)), df['type1_2013'], df['PAcrops_list_2013'] )
        df['PAcrops_list_2014'] = np.where(((df['PAcrops_list_2014'] == 0)), df['type1_2014'], df['PAcrops_list_2014'])
        df['PAcrops_list_2015'] = np.where(((df['PAcrops_list_2015'] == 0)), df['type1_2015'], df['PAcrops_list_2015'])
        df['PAcrops_list_2016'] = np.where(((df['PAcrops_list_2017'] == 0)), df['type1_2017'], df['PAcrops_list_2017'])
        df['PAcrops_list_2018'] = np.where(((df['PAcrops_list_2018'] == 0)), df['type1_2018'], df['PAcrops_list_2018'])
        df['PAcrops_list_2019'] = np.where(((df['PAcrops_list_2019'] == 0)), df['type1_2019'], df['PAcrops_list_2019'])
        df['PAcrops_list_2020'] = np.where(((df['PAcrops_list_2020'] == 0)), df['type1_2020'], df['PAcrops_list_2020'])
        df['PAcrops_list_2021'] = np.where(((df['PAcrops_list_2021'] == 0)), df['type1_2021'], df['PAcrops_list_2021'])
        df['PAcrops_list_2022'] = np.where(((df['PAcrops_list_2022'] == 0)), df['type1_2022'], df['PAcrops_list_2022'])

        df.to_csv(r"C:\Users\User\Documents\Forecast2023\Data2023\CI_dataset_replaced_raw2.csv")
        # df2 = df.loc[(df['type1_%_2013']<50)|
        #      (df['type1_%_2014']<50)|(df['type1_%_2015']<50)|(df['type1_%_2016']<50)|(df['type1_%_2017']<50)|
        #      (df['type1_%_2018']<50)|(df['type1_%_2019']<50)|(df['type1_%_2020']<50)|
        #     (df['type1_%_2021']<50)|(df['type1_%_2022']<50)]
        # df3 = df[~df.apply(tuple,1).isin(df2.apply(tuple,1))]
        m = df[['commonland', 'statecode', 'countycode','PAcrops_list_2013', 'PAcrops_list_2014', 'PAcrops_list_2015', 'PAcrops_list_2016',
            'PAcrops_list_2017', 'PAcrops_list_2018', 'PAcrops_list_2019', 'PAcrops_list_2020', 'PAcrops_list_2021','PAcrops_list_2022']]
        
              
        for y in yeas_list:
            year=str(y)
            def filter(x):
                y = str(x).strip()
                if (y == 'Soybeans')or(y =='Dbl Crop WinWht/Soybeans')or(y == 'Soybeans/Dbl Crop WinWht')or(y == 'SOYBEANS'):
                    return '1'
                if (y == 'Corn')or(y =='Dbl Crop WinWht/Corn')or(y == 'CORN'):
                    return '2'
                if(y == 'Cotton')or(y =='Dbl Crop WinWht/Cotton')or(y == 'COTTON'):
                    return '3'
                if(y == 'Spring Wheat')or(y =='Durum Wheat') or (y == 'WHEAT'):
                    return '4'
                else:
                    return '0'
            
            m[year] =m['PAcrops_list_'+ year].apply(filter)
  
        print(df.shape,m.shape)
        return m    
              

  



