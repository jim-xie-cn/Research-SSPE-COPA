import os,json,sys,logging
sys.path.append("./share")
sys.path.append("./common")
import pandas as pd
import json
from tqdm import tqdm
tqdm.pandas()
from IoTCommon import CIoTCommon
from IoTTotalFeature import CIoTTotalFeature
from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table,Float,Text
from sqlalchemy.dialects.mysql import MEDIUMTEXT, LONGTEXT
from SHMysql import CPandasMysql
from Config import g_data_root
import numpy as np
g_raw_json = "%sdataset"%g_data_root

class CIoTDbFeatures:
    def __init__(self):
        self.m_db = CPandasMysql()
        self.m_features = []
        self.m_type = {}
    def init_features(self):
        df_features = pd.read_excel("%s/features/features.xlsx"%g_data_root,index_col=0).reset_index()
        df_features = df_features[df_features['effective'] !="N"]
        df_features = df_features[df_features['count'] > 1].reset_index(drop=True)
        totalFeature = CIoTTotalFeature()
        df_features = CIoTTotalFeature.get_expand_features(df_features,totalFeature)
        df_features = df_features.rename(columns={'index': 'id'})
        self.m_db.execute("delete from feature")
        self.m_db.save("feature",df_features,action='replace')
    def get_features(self):
        return self.m_features
    def get_type(self,feature):
        return self.m_type[feature]
    def load(self):
        df_features = self.m_db.load("select * from feature")
        self.m_features = df_features['feature'].unique().tolist()
        self.m_features.extend(["protocol","Label"])
        self.m_type = {}
        self.m_type['Label']='int'
        self.m_type['protocol'] = "string"
        for i,row in df_features.iterrows():
            key = row['feature']
            type = row['type']
            self.m_type[key] = type
        return df_features

class CIoTDbSample:
    def __init__(self,trafficType):
        self.m_db = CPandasMysql()
        self.m_features = CIoTDbFeatures()
        self.m_table = CIoTDbSample.get_table_name(trafficType)
        self.m_features.load()

    def __del__(self):
        self.m_db.close()

    @staticmethod
    def get_table_name(trafficType):
        return trafficType.replace(" ","_").replace("+","").replace("(","").replace(")","").replace("__","").strip()
       
    def create_table(self,withID=True):
        self.m_db.execute("drop table IF EXISTS %s"%self.m_table)
        metadata = MetaData()
        tmp = []
        if withID:
            tmp.append(Column('id', Integer, primary_key=True, autoincrement=True))
        for key in self.m_features.get_features():
            type = self.m_features.get_type(key)
            if type in ['int']:
                tmp.append(Column(key, Integer))
            elif type in ['string','category','binary','datetime','bool']:
                tmp.append(Column(key, MEDIUMTEXT))
            elif type in ['float']:
                tmp.append(Column(key, Float))
            else:
                print("unknow feature",key,type)
        table = Table(self.m_table, metadata, *tmp)
        metadata.create_all(self.m_db.m_engine)
    
    def clear_sample(self):
        self.m_db.execute("delete from %s"%self.m_table)
    
    def import_data(self,df_data):
        try:
            self.m_db.execute("drop table IF EXISTS %s"%self.m_table)
        except:
            pass
        self.m_db.save(self.m_table,df_data,action='append')
    
    def load_data(self):
        sql = "select * from `%s`"%(self.m_table)
        return self.m_db.load(sql)
        
    def import_score(self,df_data):
        df_sample = df_data.copy(deep =True)
        #def convert_list(value):
        #    return str(value) if isinstance(value, list) else value
        #for key in tqdm(df_sample.keys().tolist(),desc="convert to json dump"):
        #    df_sample[key] = df_sample[key].apply(convert_list)
        try:
            self.m_db.execute("drop table IF EXISTS %s"%self.m_table)
        except:
            pass
        self.m_db.save(self.m_table,df_sample,action='append')
        
    def load_score(self):
        sql = "select * from `%s` order by `time` asc"%(self.m_table)
        return self.m_db.load(sql)

    def import_sample(self,df_data):
        to_drop_columns = []
        for col in df_data.keys().tolist():
            if not col in self.m_features.get_features():
               to_drop_columns.append(col)
        df_data = df_data.drop(to_drop_columns, axis=1,errors='ignore')
        self.m_db.save(self.m_table,df_data,action='append')

    def load_by_id_list(self,idList):
        total = str(idList).replace("[","(").replace("]",")")
        sql = "select * from `%s` where id in %s order by `frame.time_utc` asc"%(self.m_table,total)
        return self.m_db.load(sql)

    def load_sample(self,protocol = None,pageID = None,page_size = 10000):
        if pageID == None:
            if protocol == None:
                sql = "select * from `%s` order by `frame.time_utc` asc"%(self.m_table)
            else:
                sql = "select * from `%s` where protocol like '%s%%' order by `frame.time_utc` asc "%(self.m_table,protocol)
        else:
            if protocol == None:
                sql = "select * from `%s` where id > %d order by `frame.time_utc` asc limit %d "%(self.m_table,pageID*page_size,page_size)
            else:
                sql = "select * from `%s` where id > %d and protocol like '%s%%' order by `frame.time_utc` asc limit %d "%(self.m_table,pageID*page_size,protocol,page_size)

        df_data = self.m_db.load(sql)
        df_data = df_data.dropna(how='all')
        df_data = df_data.dropna(axis=1, how='all').reset_index(drop=True)
        return df_data

def import_sample():
    total_traffic = {}
    for fi in CIoTCommon.get_json_files(g_raw_json):
        fi = fi.replace("\\","/")
        tmp = fi.split("/")
        attackType,trafficType,fileName = tmp[-3],tmp[-2],tmp[-1].split(".")[0]
        if not trafficType in total_traffic:
            total_traffic[trafficType] = []
        total_traffic[trafficType].append(fi)

    CIoTDbFeatures().init_features()
    totalFeature = CIoTTotalFeature()

    for trafficType in tqdm(total_traffic):
        dbSample = CIoTDbSample(trafficType)
        dbSample.create_table()
        dbSample.clear_sample()
        for file in tqdm(total_traffic[trafficType],desc=trafficType):
            df_sample = pd.read_json(file)
            df_sample = df_sample.rename(columns={"frame.protocols":"protocol"})
            df_sample = df_sample[df_sample['protocol'].isin(totalFeature.get_protocols())]
            #df_sample = df_sample.drop_duplicates()
            df_sample['frame.time_utc'] = pd.to_datetime(df_sample['frame.time_utc']).dt.strftime('%Y-%m-%d %H:%M:%S.%f')
            df_sample = df_sample.reset_index(drop=True)
            dbSample.import_sample(df_sample)

def import_filter_samples(attack,interval_seconds = 3600):
    df_raw = CIoTDbSample(attack).load_sample()
    df_raw['frame.time_utc'] = pd.to_datetime(df_raw['frame.time_utc'])
    
    startTime,endTime = df_raw['frame.time_utc'].min(),df_raw['frame.time_utc'].max()
    date_rng = pd.date_range(start=startTime, end=endTime, freq='%dS'%interval_seconds)
    df_time = pd.DataFrame({"start":date_rng})
    df_time['end'] = df_time['start'] + pd.Timedelta(seconds=interval_seconds)
    def data_count(x, df_data):
        start = x['start']
        end = x['end']
        mask = (df_data['frame.time_utc'] >= start)&(df_data['frame.time_utc']<=end)
        return df_data[mask].shape[0]
        
    df_time['count'] = df_time.progress_apply(data_count,df_data=df_raw,axis=1)
    max_index = df_time['count'].idxmax()
    start = df_time.iloc[max_index]['start']
    end = df_time.iloc[max_index]['end']
    mask = (df_raw['frame.time_utc'] >= start)&(df_raw['frame.time_utc']<=end)
    df_sample = df_raw[mask].reset_index(drop=True)
    
    trafficType = "sample_%s"%attack
    dbSample = CIoTDbSample(trafficType)
    dbSample.create_table()
    dbSample.clear_sample()
    
    df_sample['frame.time_utc'] = pd.to_datetime(df_sample['frame.time_utc']).dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    dbSample.import_sample(df_sample)
    
def main():
    from IoTSample import CIoTSample
    #import_sample()
    #只对这些流量较多的时间段，进行算分等处理
    for attack in tqdm(CIoTSample().get_attack_type()):
        import_filter_samples(attack,interval_seconds = 3600)
    
def test(): 
    trafficType = "OS Fingerprinting attack"
    dbSample = CIoTDbSample(trafficType)
    df_test = dbSample.load_sample(protocol='eth:ethertype:arp',pageID=0,page_size=100)
    print(df_test)
    df_test = dbSample.load_sample(pageID=0)
    print(df_test)
    df_test = dbSample.load_sample()
    print(df_test)

if __name__ == "__main__":
    main()
