import os,json,sys,logging
sys.path.append("./share")
sys.path.append("./common")
import pandas as pd
import json
import numpy as np
from tqdm.notebook import tqdm
from IoTCommon import CIoTCommon
from IoTTotalFeature import CIoTTotalFeature
from IoTDatabase import CIoTDbSample
from IoTNoise import CIoTNoise
from Config import g_data_root,g_resolved_columns
g_sample_root = "%ssample"%g_data_root

class CIoTSample:
        
    def __init__(self):
        self.m_index = self.get_index()
        df_features = pd.read_excel("%s/features/sample-features.xlsx"%g_data_root,index_col=0).reset_index()
        df_features = df_features[df_features['effective'] !="N"]
        df_features = df_features[df_features['count'] > 1].reset_index(drop=True)
        totalFeature = CIoTTotalFeature()
        self.m_df_features = CIoTTotalFeature.get_expand_features(df_features,totalFeature)

    def get_type(self,feature):
        mask   = self.m_df_features['feature'] == feature 
        df_tmp = self.m_df_features[mask]
        if df_tmp.shape[0] <= 0: return None
        type = df_tmp.iloc[0]['type']
        values = df_tmp.iloc[0]['values']
        return type
       
    def get_index(self):
       return {'Attack': ['Port Scanning attack','Vulnerability scanner attack','Password attacks','Uploading attack','DDoS UDP Flood Attacks','DDoS ICMP Flood Attacks','DDoS TCP SYN Flood Attacks','Ransomware attack','DDoS HTTP Flood Attacks','Backdoor_attack','MITM (ARP spoofing + DNS) Attack','SQL injection attack','XSS attacks','OS Fingerprinting attack'],
        'Normal': ['IR_Receiver','Water_Level','Distance','Heart_Rate','Sound_Sensor','Soil_Moisture','phValue','Flame_Sensor','Modbus','Temperature_and_Humidity']}

    def get_attack_type(self):
        return self.m_index['Attack']

    def get_sensor_type(self):
        return self.m_index['Normal']
        
    def get_sample_by_id_list(self,trafficType,is_normal_lsit,idList, withFormat=True):
        normal_id = []
        attack_id = []
        for isNormal,id in zip(is_normal_lsit,idList):
            if isNormal:
                normal_id.append(id)
            else:
                attack_id.append(id)
        if len(normal_id)>0:
            df_normal = CIoTDbSample("Normal").load_by_id_list(normal_id)
        else:
            df_normal = pd.DataFrame()
        if len(attack_id)>0:
            df_attack = CIoTDbSample(trafficType).load_by_id_list(attack_id)
        else:
            df_attack = pd.DataFrame()
        df_sample = pd.DataFrame()
        if df_attack.shape[0] > 0:
            df_sample = pd.concat([df_sample,df_attack],ignore_index=True)
        if df_normal.shape[0] > 0:
            df_sample = pd.concat([df_sample,df_normal],ignore_index=True)
        df_sample = df_sample.reset_index(drop = True)
        if withFormat:
            df_sample = self.format(df_sample)
        else:
            to_drop_columns = []
            df_data = df_sample.copy(deep=True)
            df_data.fillna(0)
            for feature in df_data.keys().tolist():
                if df_data[feature].nunique() <= 1:
                    to_drop_columns.append(feature)
            if to_drop_columns:
                df_sample = df_sample.drop(to_drop_columns, axis=1,errors='ignore')
        
        return df_sample
    
    def get_attack_sample(self,trafficType ,protocol = None, withFormat=True,pageID=None,filtered=False):
        if filtered :
            trafficType1 = "sample_%s"%trafficType
        else:
            trafficType1 = trafficType
        df_sample = CIoTDbSample(trafficType1).load_sample(protocol,pageID=pageID)
        if withFormat:
            df_sample = self.format(df_sample)
        else:
            to_drop_columns = []
            df_data = df_sample.copy(deep=True)
            df_data.fillna(0)
            for feature in df_data.keys().tolist():
                if df_data[feature].nunique() <= 1:
                    to_drop_columns.append(feature)
            if to_drop_columns:
                df_sample = df_sample.drop(to_drop_columns, axis=1,errors='ignore')
        
        return df_sample

    def get_sensor_sample(self,trafficType,protocol=None,withFormat=True,pageID=None):
        df_result = self.get_attack_sample(trafficType,protocol,withFormat=withFormat,pageID=pageID)
        df_result['Label'] = 0
        return df_result

    #格式化数据，
    #根据类型，填充空值
    #转换为设置的类型，目前支持int,float,string,binary,datetime,bool,category
    #将类别，转换为one-hot
    def format(self,df_input):
        df_data = df_input.copy(deep = True)
        for feature in df_data.keys().tolist():
            if feature in g_resolved_columns:
                continue
            type = self.get_type(feature)
            if type == 'int':
                df_data[feature] = df_data[feature].replace("", np.nan)
                df_data[feature] = df_data[feature].fillna(0)
                df_data[feature] = df_data[feature].astype('float')
                df_data[feature] = df_data[feature].astype('int')
            elif type == 'string':
                df_data[feature] = df_data[feature].replace("", np.nan)
                df_data[feature] = df_data[feature].fillna("")
                df_data[feature] = df_data[feature].astype('str')
            elif type == 'binary':
                df_data[feature] = df_data[feature].replace("", np.nan)
                df_data[feature] = df_data[feature].fillna("")
                df_data[feature] = df_data[feature].astype('str')
            elif type == 'float':
                df_data[feature] = df_data[feature].replace("", np.nan)
                df_data[feature] = df_data[feature].fillna(0.0)
                df_data[feature] = df_data[feature].astype('float')
            elif type == 'category':
                df_data[feature] = df_data[feature].astype('str')
                df_data[feature] = df_data[feature].replace("", np.nan)
                df_data[feature] = df_data[feature].fillna("")
                df_data[feature] = df_data[feature].astype('category')
            elif type == 'datetime':
                df_data[feature] = df_data[feature].replace("", np.nan)
                df_data[feature] = df_data[feature].fillna(0)
                df_data[feature] = pd.to_datetime(df_data[feature])
            elif type == 'bool':
                if not CIoTCommon.is_number(df_data[feature]):
                    df_data[feature].replace("", np.nan, inplace=True)
                    df_data[feature] = df_data[feature].notna()
                else:
                    df_data[feature] = df_data[feature].replace("", np.nan)
                df_data[feature] = df_data[feature].fillna(False)
                df_data[feature] = df_data[feature].astype('float')
                df_data[feature] = df_data[feature].astype('int')
                df_data[feature] = df_data[feature].astype('bool')
            else:
                print("Unknow feature type",feature,type)

        to_drop_columns = []
        for feature in df_data.keys().tolist():
            if feature in g_resolved_columns:
                continue
            if df_data[feature].nunique() <= 1:
                to_drop_columns.append(feature)
        if to_drop_columns:
            df_data = df_data.drop(to_drop_columns, axis=1,errors='ignore')
        return df_data
    
    def format_flat(self,df_input):
        df_data = df_input.copy(deep = True)
        for feature in df_data.keys().tolist():
            if feature in g_resolved_columns:
                continue
            raw_feature = feature.split("_FLAT_")[0]
            type = self.get_type(raw_feature)
            if type == 'int':
                df_data[feature] = df_data[feature].replace("", np.nan)
                df_data[feature] = df_data[feature].fillna(0)
                df_data[feature] = df_data[feature].astype('float')
                df_data[feature] = df_data[feature].astype('int')
            elif type == 'string':
                df_data[feature] = df_data[feature].replace("", np.nan)
                df_data[feature] = df_data[feature].fillna("")
                df_data[feature] = df_data[feature].astype('str')
            elif type == 'binary':
                df_data[feature] = df_data[feature].replace("", np.nan)
                df_data[feature] = df_data[feature].fillna("")
                df_data[feature] = df_data[feature].astype('str')
            elif type == 'float':
                df_data[feature] = df_data[feature].replace("", np.nan)
                df_data[feature] = df_data[feature].fillna(0.0)
                df_data[feature] = df_data[feature].astype('float')
            elif type == 'category':
                df_data[feature] = df_data[feature].astype('str')
                df_data[feature] = df_data[feature].replace("", np.nan)
                df_data[feature] = df_data[feature].fillna("")
                df_data[feature] = df_data[feature].astype('category')
            elif type == 'datetime':
                df_data[feature] = df_data[feature].replace("", np.nan)
                df_data[feature] = df_data[feature].fillna(0)
                df_data[feature] = pd.to_datetime(df_data[feature])
            elif type == 'bool':
                if not CIoTCommon.is_number(df_data[feature]):
                    df_data[feature].replace("", np.nan, inplace=True)
                    df_data[feature] = df_data[feature].notna()
                else:
                    df_data[feature] = df_data[feature].replace("", np.nan)
                df_data[feature] = df_data[feature].fillna(False)
                df_data[feature] = df_data[feature].astype('float')
                df_data[feature] = df_data[feature].astype('int')
                df_data[feature] = df_data[feature].astype('bool')
            else:
                print("Unknow feature type",feature,type)

        to_drop_columns = []
        for feature in df_data.keys().tolist():
            if feature in g_resolved_columns:
                continue
            if df_data[feature].nunique() <= 1:
                to_drop_columns.append(feature)
        if to_drop_columns:
            df_data = df_data.drop(to_drop_columns, axis=1,errors='ignore')
        return df_data
    
    '''
    添加噪音
    '''
    def add_noise(self,df_input,ratio=0.1):
        df_data = df_input.copy(deep = True)
        for feature in df_data.keys().tolist():
            if feature in g_resolved_columns:
                continue
            type = self.get_type(feature)
            if type == None:
                dtypes_dict = df_input.dtypes.to_dict()
                type = dtypes_dict[feature]
            df_data[feature] = CIoTNoise.add_noise(feature,type,df_data,ratio)
        return df_data
        
    def add_noise_flat(self,df_input,ratio=0.1):
        df_data = df_input.copy(deep = True)
        for feature in df_data.keys().tolist():
            if feature in g_resolved_columns:
                continue
            raw_feature = feature.split("_FLAT_")[0]
            type = self.get_type(raw_feature)
            if type == None:
                dtypes_dict = df_input.dtypes.to_dict()
                type = dtypes_dict[feature]
            df_data[feature] = CIoTNoise.add_noise(feature,type,df_data,ratio)
        return df_data

def main():
    test = CIoTSample()
    trafficType = "Backdoor_attack"
    for trafficType in test.get_attack_type():
        print(trafficType)
        protocol = "eth:ethertype:ip:udp:dns"
        df_data = test.get_attack_sample(trafficType,protocol)
        print(df_data.dtypes)
        #object_columns = df_data.select_dtypes(include=['object'])
        #print(object_columns)
        df_noised = test.add_noise(df_data)
        #print(df_data.mean())
        #print(df_noised.mean())

if __name__ == "__main__":
    main()
