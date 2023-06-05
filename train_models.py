from collections import defaultdict
import glob
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt

class train_model(object):
    def __init__(self, acreage_data, crop_id_data, planting_dates_data):
        self.acreage_data = acreage_data
        self.crop_id_data = crop_id_data
        self.planting_dates_data = planting_dates_data

    def CI_report(self, df_pred, year_list): 
        df =  df_pred[['commonland',  'state_code', 'county_code','2013', '2014', '2015',
                         '2016', '2017', '2018', '2019', '2020', '2021', '2022', 'predictions']]
                
        def filter(x):
            y = str(x).strip()
            if (y == '1'):
                return 'Soybeans'
            if (y == '2'):
                return 'Corn'
            if(y == '3' ):
                return 'Cotton'
            if(y == '4'):
                return  'Wheat'
            else:
                return 'Other'
            
        for year in year_list:

            df[year] = df[ year].apply(filter)          
                    
        df ['predictions'] = df['predictions'].apply(filter).astype(str) 
        def code(x):
            y = str(x).strip()
            if (y == 'Soybeans'):
                return str('0081')
            if (y == 'Corn'):
                return str('0041')
            if(y == 'Cotton' ):     
                return str('0021')
            if(y == 'Wheat'):
                return str('0011' )
            else:
                return str('9100' )
                
        df ['Pred_Proag_Code'] = df['predictions'].apply(code)  
        return df

    def Crop_Id_RF(self, modeldata ) :
         
         mtx = pd.read_csv(self.crop_id_data)
         X = mtx[['statecode','countycode','2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']].to_numpy(dtype='float')
         y = mtx[['2021']].to_numpy(dtype='float')
         X_eval = mtx[['statecode','countycode', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']].to_numpy(dtype='float')
         y_eval = mtx[['2022']].to_numpy(dtype='float')

         #balancing data
         strategy_under = {0.0: 80000, 1.0: 85764, 2.0:80000, 3.0:80000, 4.0: 60000}# put here the minority cklass with initial values of mayority
         over = RandomOverSampler()
         under = RandomUnderSampler(sampling_strategy=strategy_under)

         X_over, y_over = over.fit_resample(X, y)
         print(f"Oversampled: {Counter(y_over)}")

         X_under, y_under = under.fit_resample(X_over, y_over)
         print(f"undersampled: {Counter(y_under)}")       

         X_train, X_test, y_train, y_test = train_test_split( X_over, y_over, test_size = 0.2, random_state = 42)
         print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)



        
         model = RandomForestClassifier(n_jobs = 8, verbose=1, class_weight = 'balanced')
         bst=model.fit(X_train, y_train)
         y_pred = bst.predict(X_test)
         predictions = [round(value) for value in y_pred]
         accuracy = accuracy_score(y_test, predictions)
         print("Accuracy: %.2f%%" % (accuracy * 100.0))
         report = classification_report(y_test, y_pred, output_dict=True)
         print(classification_report(y_test, y_pred))

        # Evaluation
       
         y_pred_eval = bst.predict(X_eval)

         predictions = [round(value) for value in y_pred_eval]
         accuracy = accuracy_score(y_eval, predictions)
         print("Accuracy: %.2f%%" % (accuracy * 100.0))
         report = classification_report(y_eval, y_pred_eval, output_dict=True)
         print(classification_report(y_eval, y_pred_eval))

         joblib.dump(bst, modeldata, compress=3)

    def Crop_Id_Xgboost(self, modeldata) :  
        mtx = pd.read_csv(self.crop_id_data)
        X = mtx[['statecode','countycode','2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']].to_numpy(dtype='float')
        y = mtx[['2022']].to_numpy(dtype='float')
        #X_eval = mtx[['statecode','countycode', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']].to_numpy(dtype='float')
        #y_eval = mtx[['2022']].to_numpy(dtype='float')
        
        X_train, X_test, y_train, y_test = train_test_split( X ,y, test_size = 0.2, random_state = 42)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)   

        model = xgb.XGBClassifier(objective='multi:softmax', num_class=5, max_depth= 7, n_jobs = -1) 
       
        bst = model.fit(X_train, y_train)

        y_pred = bst.predict(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        report = classification_report(y_test, y_pred, output_dict=True)
        print(classification_report(y_test, y_pred))

        # y_pred_eval = bst.predict(X_eval)

        # predictions = [round(value) for value in y_pred_eval]
        # accuracy = accuracy_score(y_eval, predictions)
        # print("Accuracy: %.2f%%" % (accuracy * 100.0))
        # report = classification_report(y_eval, y_pred_eval, output_dict=True)
        # print(classification_report(y_eval, y_pred_eval))

        joblib.dump(bst, modeldata, compress=3)
        

        #mtx['predictions'] = y_pred_eval

        return mtx



    def Crop_Id_LSTM(self) :     

        mtx = pd.read_csv(self.crop_id_data)
        X = mtx[['statecode','countycode','2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']].to_numpy(dtype='float')
        y = mtx[['2021']].to_numpy(dtype='float')
        X_eval = mtx[['statecode','countycode', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']].to_numpy(dtype='float')
        y_eval = mtx[['2022']].to_numpy(dtype='float')

        X_train, X_test, y_train, y_test = train_test_split( X ,y, test_size = 0.2, random_state = 42)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape) 
        max_review_length = 11
        X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
        X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

        
        model = Sequential()
        model.add()
        model.add(LSTM(100))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(X_train, y_train, epochs=3, batch_size=64)
        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))


    def Acreage_Forecast(self) :

        d = pd.read_csv(self.planting_dates_data)  
        d['median']=d.filter(like='20').apply(lambda x: x.median(), axis=1) 
        d['std']=d.filter(like='20').apply(lambda x: x.std(), axis=1)   
        d["distance"] = d['area'] - d['median']
        d['median'] = np.where((d['distance'] <0), (d['median']-d['std']), d['median'] )
       
        return d

    def PD_Xgboost(self, modeldata) :  
        mtx = pd.read_csv(self.planting_dates_data)
        X = mtx[[ 'Delta_2014','Delta_2015', 'Delta_2016', 'Delta_2017', 'Delta_2018',
       'Delta_2019', 'Delta_2020', 'Delta_2021']].to_numpy(dtype='float')
        y = mtx[['Delta_2022']].to_numpy(dtype='float')

    #     X_eval =mtx[['Delta_2015', 'Delta_2016', 'Delta_2017', 'Delta_2018', 'Delta_2019',
    #    'Delta_2020', 'Delta_2021']].to_numpy(dtype='float')
    #     #X_eval = mtx[['statecode','countycode', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']].to_numpy(dtype='float')
    #     y_eval = mtx[['Delta_2022']].to_numpy(dtype='float')
        
        X_train, X_test, y_train, y_test = train_test_split( X ,y, test_size = 0.2, random_state = 42)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)  

        regressor=xgb.XGBRegressor(eval_metric='rmse')
        # set up our search grid
        param_grid = {"max_depth":    [4, 5, 6],
              "n_estimators": [500, 600, 700],
              "learning_rate": [0.01, 0.015]}

        # try out every combination of the above values
        search = GridSearchCV(regressor, param_grid, cv=5).fit(X_train, y_train)

        print("The best hyperparameters are ",search.best_params_) 

        regressor=xgb.XGBRegressor(learning_rate = search.best_params_["learning_rate"],
                           n_estimators  = search.best_params_["n_estimators"],
                           max_depth     = search.best_params_["max_depth"],
                           eval_metric='rmsle')

        regressor.fit(X_train, y_train)
        predictions = regressor.predict(X_test)

        RMSE = np.sqrt( mean_squared_error(y_test, predictions) )
        print("The score is %.5f" % RMSE )
        joblib.dump(regressor, modeldata, compress=3)

        # predictions_validation = regressor.predict(X_eval)

        # RMSE = np.sqrt( mean_squared_error(y_eval, predictions_validation) )
        # print("Evaluation %.5f" % RMSE )
        # mtx['predictions'] = predictions_validation

        return mtx
    
    def PD_LSTM(self) :
        mtx = pd.read_csv(self.planting_dates_data)
        X = mtx[['Delta_2014', 'Delta_2015', 'Delta_2016', 'Delta_2017', 'Delta_2018',
       'Delta_2019', 'Delta_2020']].to_numpy(dtype='float')
        y = mtx[['Delta_2021']].to_numpy(dtype='float')

        X_eval =mtx[['Delta_2015', 'Delta_2016', 'Delta_2017', 'Delta_2018', 'Delta_2019',
       'Delta_2020', 'Delta_2021']].to_numpy(dtype='float')
        #X_eval = mtx[['statecode','countycode', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']].to_numpy(dtype='float')
        y_eval = mtx[['Delta_2022']].to_numpy(dtype='float')
        
        X_train, X_test, y_train, y_test = train_test_split( X ,y, test_size = 0.2, random_state = 42)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape) 

        model = tf.keras.Sequential()
        model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    units=128, 
                    input_shape=(7, 1) #X_train.shape[0], X_train.shape[1])
                )
            )
        )
        model.add(keras.layers.Dropout(rate=0.3))
        model.add(keras.layers.Dense(units=2))
        model.compile(loss='mean_squared_error', optimizer='adam',  metrics=['mean_squared_error'])
        history = model.fit(   
            X_train, y_train, 
            epochs=40, 
            batch_size=24, 
            validation_split=0.1,
    
            shuffle=False
        )
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend();    
        y_pred = model.predict(X_test)

        RMSE = np.sqrt( mean_squared_error(y_test, y_pred) )
        print("The score is %.5f" % RMSE )
        #joblib.dump(regressor, modeldata, compress=3)

        predictions_validation = regressor.predict(X_eval)

        RMSE = np.sqrt( mean_squared_error(y_eval, predictions_validation) )
        print("Evaluation %.5f" % RMSE )
        mtx['predictions'] = predictions_validation
        
        return mtx

