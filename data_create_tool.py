import gzip
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
import impyute.imputation.ts as ts
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pyts.image import RecurrencePlot
from pyts.image import MarkovTransitionField
from pyts.multivariate.image import JointRecurrencePlot
from sklearn.impute import KNNImputer

def load_area_split(df):        
    data = df
    data = data.sort_values(['AREA_INDEX', 'TIME_INDEX'])
    data.index = range(data['AREA_INDEX'].size)
    area = data['AREA_INDEX']
    index = area[~area.duplicated(keep='first')].index
    area = area.drop_duplicates()
    area = area.to_numpy()
    result=dict()
    for idx in range(index.size - 1):
        result[area[idx]] = data.loc[index[idx]: index[idx + 1] - 1]
    result[area[area.size - 1]] = data.loc[index[index.size - 1]: data['AREA_INDEX'].size - 1]
    return result


def markov_transition(data,img_size = 24):
    X = data
    mtf = MarkovTransitionField(image_size=img_size)
    X_mtf = mtf.fit_transform(X)
    return X_mtf
    

def recurrence_plot(data,t_hold='point',percent = 50,w_size =24):
    rp = RecurrencePlot(threshold=t_hold, percentage=percent)
    X = data.reshape(-1,w_size)
    X_rp = rp.fit_transform(X)
    return X_rp


def joint_recurrence_plot(df,threshold_val,percentage_val):
    """[summary]
    Input: dataframe 

    Args:
        df ([type]): [description]
    """
    result = df.copy()
    for area in tqdm_notebook(result.keys(),desc = "to image"):
        X = result[area]
        X[np.isnan(X)] = -1
        X = X.to_numpy()

        arr = np.array(X[0:24].T)
        # arr = np.append(arr,arr,axis=0)
        a = int(len(X)/24)
        for col in range(1,a):
            # print(col)
            arr = np.append(arr,X[24*col:24*col + 24].T,axis=0)
        X = arr.reshape(-1,9,24)
        # Recurrence plot transformation
        jrp = JointRecurrencePlot(threshold=threshold_val, percentage=percentage_val)
        X_jrp = jrp.fit_transform(X)
        result[area] = X_jrp
        
    return result
    

def KNN_imputer(df,train,n_neighbour):
    data = tf_error_np(df)
    imputer = KNNImputer(n_neighbors=n_neighbour)
    imputer.fit(train)
    data = imputer.transform(data)
    return data

def tf_error_np(df)->pd.DataFrame:
        """[summary]
            columns value = np.nan that FLAG != 1 (anomaly data)
        Returns:
            pd.DataFrame: FLAG != 1 value -> np.nan
        """
        result = df.copy()
        for col in tqdm_notebook(['SO2','NOX','NO','NO2','O3','CO','PM10','PM25']):
            flag = col + '_FLAG'
            result[col].iloc[np.where(result[flag] != 1)] = np.nan
        return result


class Data_split():
    def __init__(self,path = None,dataframe= None ,Mode = "path"):
        if Mode == "path":
            with gzip.open(path,'rb') as f:
                self.df = pickle.load(f)
        else:
            self.df = dataframe

    # return list or dict data which is splited by month
    def load_month_split(self ,return_type='list'):
        data = self.df.copy()
        date = data['MDATETIME'].str.slice(start=0, stop=7)
        index = date[~date.duplicated(keep='first')].index
        date = date.drop_duplicates()
        date = date.str.replace('/', '')
        date = date.astype('int64')
        date = date.to_numpy()
        if(return_type == 'list'):
            result = list()
            for idx in range(index.size - 1):
                result.append(data.loc[index[idx]: index[idx + 1] - 1])
            result.append(data.loc[index[index.size - 1]: data['AREA_INDEX'].size - 1])
            return result
        elif(return_type == 'dict'):
            result = dict()
            for idx in range(index.size - 1):
                result[date[idx]] = data.loc[index[idx]: index[idx + 1] - 1]
            result[date[date.size - 1]] = data.loc[index[index.size - 1]: data['AREA_INDEX'].size - 1]
            return result
        return -1

    # return list or dict data which is splited by day
    def load_day_split(self ,return_type = 'list'):
        data = self.df.copy()
        date = data['MDATETIME'].str.slice(start=0, stop=10)
        index = date[~date.duplicated(keep='first')].index
        date = date.drop_duplicates()
        date = date.str.replace('/', '')
        date = date.astype('int64')
        date = date.to_numpy()
        if(return_type == 'list'):
            result = list()
            for idx in range(index.size - 1):
                result.append(data.loc[index[idx]: index[idx + 1] - 1])
            result.append(data.loc[index[index.size - 1]: data['AREA_INDEX'].size - 1])
            return result
        elif(return_type == 'dict'):
            result = dict()
            for idx in range(index.size - 1):
                result[date[idx]] = data.loc[index[idx]: index[idx + 1] - 1]
            result[date[date.size - 1]] = data.loc[index[index.size - 1]: data['AREA_INDEX'].size - 1]
            return result
        return -1

    # return list or dict data which is splited by hour
    # if call with return_type = 'list', return data without column'TIME_INDEX'
    def load_hour_split(self,return_type = 'list'):
        data = self.df.copy()
        date = data['MDATETIME'].str.slice(start=0, stop=13)
        index = date[~date.duplicated(keep='first')].index
        date = date.drop_duplicates()
        date = date.str.replace('/', '')
        date = date.str.replace(' ', '')
        date = date.astype('int64')
        date = date.to_numpy()
        if(return_type == 'list'):
            result = list()
            data = data.drop(['TIME_INDEX'], axis=1)
            for idx in range(index.size - 1):
                result.append(data.loc[index[idx]: index[idx + 1] - 1])
            result.append(data.loc[index[index.size - 1]: data['AREA_INDEX'].size - 1])
            return result
        elif(return_type == 'dict'):
            result = dict()
            for idx in range(index.size - 1):
                result[date[idx]] = data.loc[index[idx]: index[idx + 1] - 1]
            result[date[date.size - 1]] = data.loc[index[index.size - 1]: data['AREA_INDEX'].size - 1]
            return result
        return -1

    def load_area_split(self,df=None):
        if df ==None:
            data = self.df.copy()
        else:
            data = df
        data = data.sort_values(['AREA_INDEX', 'TIME_INDEX'])
        data.index = range(data['AREA_INDEX'].size)
        area = data['AREA_INDEX']
        index = area[~area.duplicated(keep='first')].index
        area = area.drop_duplicates()
        #area = area.astype('int64')
        area = area.to_numpy()
        result=dict()
        for idx in range(index.size - 1):
            result[area[idx]] = data.loc[index[idx]: index[idx + 1] - 1]
        result[area[area.size - 1]] = data.loc[index[index.size - 1]: data['AREA_INDEX'].size - 1]
        return result

    def show_data(self):
        print(self.df)

    def __del__(self):
        pass


class Data_imputation_tool():
    def __init__(self,df:pd.DataFrame=None,path:str=None,Mode:str='df'):
        if Mode =='df':
            self.df =df
        elif Mode =='path':
            with gzip.open(path, 'rb') as f:
                self.df = pickle.load(f)
        elif Mode =='tool':
            self.df = None
        else:
            print("No Mode")
            exit()
    
    def tf_error_np(self)->pd.DataFrame:
        """[summary]
            columns value = np.nan that FLAG != 1 (anomaly data)
        Returns:
            pd.DataFrame: FLAG != 1 value -> np.nan
        """
        result = self.df.copy()
        for col in tqdm_notebook(['SO2','NOX','NO','NO2','O3','CO','PM10','PM25']):
            flag = col + '_FLAG'
            result[col].iloc[np.where(result[flag] != 1)] = np.nan
        return result
    

    def load_area_split(self,df =None):
        if df == None:
            data = self.df.copy()
        else:
            data = df
        

        data = data.sort_values(['AREA_INDEX', 'TIME_INDEX'])
        data.index = range(data['AREA_INDEX'].size)
        area = data['AREA_INDEX']
        index = area[~area.duplicated(keep='first')].index
        area = area.drop_duplicates()
        #area = area.astype('int64')
        area = area.to_numpy()
        result=dict()
        for idx in range(index.size - 1):
            result[area[idx]] = data.loc[index[idx]: index[idx + 1] - 1]
        result[area[area.size - 1]] = data.loc[index[index.size - 1]: data['AREA_INDEX'].size - 1]
        return result

    def area_loc(self,df):
        locf_data = df.copy()
        for area in tqdm_notebook(locf_data.keys()):
            try:
                locf_data[area] = locf_data[area].to_numpy()
                locf_data[area]= ts.locf(locf_data[area],axis=1)
            except:
                print(area)
        return locf_data

    def del_error_data(self,df):
        sel_col = ['TIME_INDEX','AREA_INDEX','RAIN','SO2','NOX','NO','NO2','O3','CO','PM10','PM25']
        need_f = ['SO2','NOX','NO','NO2','O3','CO','PM10','PM25']
        for col in need_f:
            sel_col.append(col+'_FLAG')
        df = df[sel_col]

        for col in tqdm_notebook(['SO2','NOX','NO','NO2','O3','CO','PM10','PM25']):
            flag = col + '_FLAG'
            df[col].iloc[np.where(tmp[flag] != 1)] = np.nan
        df.dropna(inplace=True)
        return df
    

    def area_move_window(self,df):
        imp_data = del_error_data(df)
        for area in tqdm_notebook(imp_data.keys()):
            imp_data[area] = imp_data[area].to_numpy()
            imp_data[area]= ts.moving_window(imp_data[area],wsize=3,axis=1)
            imp_data[area] = pd.DataFrame(data=imp_data[area],columns=['TIME_INDEX', 'AREA_INDEX', 'RAIN', 'SO2', 'NOX', 'NO', 'NO2', 'O3',
            'CO', 'PM10', 'PM25', 'SO2_FLAG', 'NOX_FLAG', 'NO_FLAG', 'NO2_FLAG',
            'O3_FLAG', 'CO_FLAG', 'PM10_FLAG', 'PM25_FLAG'])
        return imp_data
    
    def area_locf(self,df):
        imp_data = del_error_data(df)
        for area in tqdm_notebook(imp_data.keys()):
            imp_data[area] = imp_data[area].to_numpy()
            imp_data[area]= ts.locf(imp_data[area])
            imp_data[area] = pd.DataFrame(data=imp_data[area],columns=['TIME_INDEX', 'AREA_INDEX', 'RAIN', 'SO2', 'NOX', 'NO', 'NO2', 'O3',
            'CO', 'PM10', 'PM25', 'SO2_FLAG', 'NOX_FLAG', 'NO_FLAG', 'NO2_FLAG',
            'O3_FLAG', 'CO_FLAG', 'PM10_FLAG', 'PM25_FLAG'])
        return imp_data
    
    def simple_imputer(self,train_df,imp_df):
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(train_df)
        imp_data = imp_df
        for area in tqdm_notebook(imp_data.keys()):
            imp_data[area] = imp_data[area].to_numpy()
            imp_data[area]= imp_mean.transform(imp_data[area])
            imp_data[area] = pd.DataFrame(data=imp_data[area],columns=['TIME_INDEX', 'AREA_INDEX', 'RAIN', 'SO2', 'NOX', 'NO', 'NO2', 'O3',
            'CO', 'PM10', 'PM25', 'SO2_FLAG', 'NOX_FLAG', 'NO_FLAG', 'NO2_FLAG',
            'O3_FLAG', 'CO_FLAG', 'PM10_FLAG', 'PM25_FLAG'])
        return imp_data