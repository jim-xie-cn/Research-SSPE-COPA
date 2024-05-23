import os,json,sys,logging
sys.path.append("./share")
sys.path.append("./common")
import pandas as pd
import json
from tqdm import tqdm
from IoTCommon import CIoTCommon
from IoTTotalFeature import CIoTTotalFeature
from Config import g_data_root
g_sample_root = "%ssample"%g_data_root

class CIoTSample:
        
    def __init__(self):
        self.m_index = self.get_index()
        df_features = pd.read_excel("%s/features/features.xlsx"%g_data_root,index_col=0).reset_index()
        df_features = df_features[df_features['effective'] !="N"]
        df_features = df_features[df_features['count'] > 1].reset_index(drop=True)
        totalFeature = CIoTTotalFeature()
        self.m_df_features = CIoTTotalFeature.get_expand_features(df_features,totalFeature)

    def get_type(self,feature):
        mask   = self.m_df_features['feature'] == feature 
        df_tmp = self.m_df_features[mask]
        if df_tmp.shape[0] <= 0: return None,None
        type = df_tmp.iloc[0]['type']
        values = df_tmp.iloc[0]['values']
        return type
       
    def get_index(self):
        all_data =[]
        for fi in CIoTCommon.get_json_files(g_sample_root):
            fi = fi.replace("\\","/")
            temp = fi.split("/")
            attackType,trafficType,segment,protocol = temp[-4],temp[-3],temp[-2],temp[-1].split(".json")[0]
            protocol = protocol.replace("-",":")
            tmp = {}
            tmp['attackType'] = attackType
            tmp['trafficType'] = trafficType
            tmp['segment'] = segment
            tmp['protocol'] = protocol
            tmp['file'] = fi
            all_data.append(tmp)
        return pd.DataFrame(all_data)

    def get_attack_type(self):
        df_tmp = self.m_index[self.m_index['attackType'] == 'Attack']
        return df_tmp['trafficType'].unique().tolist()

    def get_sensor_type(self):
        df_tmp = self.m_index[self.m_index['attackType'] == 'Normal']
        return df_tmp['trafficType'].unique().tolist()
    
    def get_attack_protocol(self,trafficType):
        mask = (self.m_index['attackType']=='Attack') & (self.m_index['trafficType']==trafficType)
        df_tmp = self.m_index[mask]
        return df_tmp['protocol'].unique().tolist()

    def get_attack_sample(self,trafficType = None,protocol = None):
        if trafficType == None:
            if protocol == None:
                mask = (self.m_index['attackType']=='Attack')
            else:
                mask = (self.m_index['attackType']=='Attack') & (self.m_index['protocol']==protocol)
        else:
            if protocol == None:
                mask = (self.m_index['attackType']=='Attack') & (self.m_index['trafficType']==trafficType)
            else:
                mask = (self.m_index['attackType']=='Attack') & (self.m_index['trafficType']==trafficType) & (self.m_index['protocol']==protocol)
        
        df_sample = pd.DataFrame()
        
        if self.m_index[mask].shape[0] <= 0:
            print("No Attack sample found trafficType=%r,protocol=%r"%(trafficType,protocol))
            return df_sample
        
        for file in self.m_index[mask]['file'].tolist():
            df_tmp = pd.read_json(file)
            df_tmp['file'] = file
            df_sample = pd.concat([df_sample,df_tmp],ignore_index=False)

        return df_sample.sort_values(by='frame.time_utc').reset_index(drop=True)

    def get_sensor_protocol(self,trafficType):
        mask = (self.m_index['attackType']=='Normal') & (self.m_index['trafficType']==trafficType)
        df_tmp = self.m_index[mask]
        return df_tmp['protocol'].unique().tolist()
    
    def get_sensor_sample(self,trafficType=None,protocol = None):
        if trafficType == None:
            if protocol == None:
                mask = (self.m_index['attackType']=='Normal')
            else:
                mask = (self.m_index['attackType']=='Normal') & (self.m_index['protocol']==protocol)
        else:
            if protocol == None:
                mask = (self.m_index['attackType']=='Normal') & (self.m_index['trafficType']==trafficType)
            else:
                mask = (self.m_index['attackType']=='Normal') & (self.m_index['trafficType']==trafficType) & (self.m_index['protocol']==protocol)
        
        df_sample = pd.DataFrame()
        
        if self.m_index[mask].shape[0] <= 0:
            print("No Normal sample found trafficType=%r,protocol=%r"%(trafficType,protocol))
            return df_sample
            
        for file in self.m_index[mask]['file'].tolist():
            df_tmp = pd.read_json(file)
            df_tmp['file'] = file
            df_sample = pd.concat([df_sample,df_tmp],ignore_index=False)
        
        return df_sample.sort_values(by='frame.time_utc').reset_index(drop=True)
            
    #格式化数据，
    #根据类型，填充空值
    #转换为设置的类型，目前支持int,float,string,binary,datetime,bool,category
    #将类别，转换为one-hot
    def format(self,df_data):
        categroy_columns = []
        for feature in df_data.keys().tolist():
            if feature in ['id','protocol','Label','file']:
                continue
            type = self.get_type(feature)
            if type == 'int':
                df_data[feature] = df_data[feature].fillna(0)
                df_data[feature] = df_data[feature].astype('int')
            elif type == 'string':
                df_data[feature] = df_data[feature].fillna("")
                df_data[feature] = df_data[feature].astype('str')
            elif type == 'binary':
                df_data[feature] = df_data[feature].fillna("")
            elif type == 'float':
                df_data[feature] = df_data[feature].fillna(0.0)
                df_data[feature] = df_data[feature].astype('float')
            elif type == 'category':
                #df_data[feature] = df_data[feature].fillna("-1")
                df_data[feature] = df_data[feature].astype('category')
                categroy_columns.append(feature)
            elif type == 'datetime':
                df_data[feature] = df_data[feature].fillna(0)
                df_data[feature] = pd.to_datetime(df_data[feature], unit='ms')
            elif type == 'bool':
                df_data[feature] = df_data[feature].fillna(False)
                df_data[feature] = df_data[feature].astype('bool')
            else:
                print("Unknow feature type",feature,type)
        
        df_encoded = pd.get_dummies(df_data, columns=categroy_columns)
        df_combined = pd.concat([df_data, df_encoded], axis=1)
        df_combined = df_combined.drop(columns=categroy_columns, axis=1)
        return df_encoded
        
def main():
    test = CIoTSample()
    trafficType = "Backdoor_attack"
    protocol = "eth:ethertype:ip:udp:dns"
    df_test = test.get_attack_sample(trafficType=trafficType,protocol=protocol)
    df_test = test.format(df_test)
    print(df_test.dtypes)
    
if __name__ == "__main__":
    main()
