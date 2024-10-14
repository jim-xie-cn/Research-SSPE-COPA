'''
1. 数据库初始化（1.将样本保存至数据库。2.随机从传感器流量中采样，生成正常流量，保存至数据库）
2. 计算攻击分数（1.将没有流量的时间点，填充为正常流量。2.使用滑动窗口计算流量）
4. ML训练时，只选择7种类型的特征。支持7种类型（int和bool保存为int、category，binary，string, datetime保存为text、float保存为Float）
5. 生成数据集（packet-label），用于ML训练：单数据包-->标签 (选择7种类型的特征)
6. 生成数据集（packet-score），用于ML训练：多数据包-->频谱 (选择7种类型的特征，将滑动窗口内的特征进行合并)
'''
import os,json,sys,logging
from numpy import nan
sys.path.append("./share")
sys.path.append("./common")
import pandas as pd
import json,logging
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from IoTCommon import CIoTCommon
from SHDataProcess import CSHDataProcess
from IoTTotalFeature import CIoTTotalFeature
from IoTDatabase import CIoTDbFeatures,CIoTDbSample
from IoTSample import CIoTSample
from IoTScore import CIoTScore,Create_Score_Sample,Create_Normal_Sample
from SHSample import CSHSample
from Config import g_data_root,get_attack_score,get_series_score,g_resolved_columns,g_sample_cfg
from sklearn.metrics import jaccard_score
from scipy.stats import skew, kurtosis
import warnings
import argparse
warnings.simplefilter("ignore")
pd.set_option('display.float_format', lambda x: '%.4f' % x)
np.set_printoptions(suppress=True)
g_sample_root = "%ssample"%g_data_root

class IoTMLSample:
    def __init__(self,attack):
        self.m_attack = attack
        self.m_measure = g_sample_cfg[attack]['measure']
        self.m_interval = g_sample_cfg[attack]['interval']
        self.m_noise = g_sample_cfg[attack]['noise']
        self.m_max_sample = g_sample_cfg[attack]['max_sample']
        self.m_train_sample = 10000

    @staticmethod
    def get_level(df_data,measure):
        df_sample = df_data.copy(deep=True)
        value_ratio = (df_sample['Label'].value_counts() ) / len(df_sample['Label'])
        df_sample['score']= df_sample[measure]
        #df_sample['score'] = (df_sample['score'] - df_sample['score'].mean()) / df_sample['score'].std()
        bins,intervals = CIoTCommon.get_level_by_score(df_sample,value_ratio)
        return bins.astype(int)
        
    @staticmethod
    def get_flat(df_data):
        flattened_data = df_data.values.flatten()
        new_columns = [f'{col}_FLAT_{i}' for i in range(len(df_data)) for col in df_data.columns]
        return pd.DataFrame([flattened_data],columns=new_columns)
        
    @staticmethod
    def select_number_features(df_sample):
        #测试：将frame.len,udp.time_delta去掉      
        #g_debug_skip_cols = ['frame.len','ip.len','data.len','tcp.len''udp.length','tcp.analysis.push_bytes_sent','tcp.srcport']
        #df_data = df_data.drop(g_debug_skip_cols, axis=1,errors='ignore')
        df_data = df_sample.drop(['id','time','frame.time_delta','normal'], axis=1,errors='ignore')
        df_data = df_data.select_dtypes(include=['float','bool','int','int64'])        
        return df_data;
        
    @staticmethod
    def get_packet_text(data):
        text = "{"
        for key, value in data.items():
            v = str(value)
            if v == np.nan or v in ['','nan','None']:
                continue
            if key in g_resolved_columns:
                continue
            if key in ['id','Label','score','frame.time_delta','normal']:
                continue
            text += f'"{key}":"{value}",'
        text = text.rstrip(",") + "}"
        return g_prompt_template.format(data=text)

    def get_packet_label(self):
        #读取原始数据，防止数据量过大
        ioTSample = CIoTSample()
        df_raw = ioTSample.get_attack_sample(self.m_attack,withFormat=True)
        df_raw.loc[df_raw['Label']==-1,"Label"]=0
        df_raw['time'] = pd.to_datetime(df_raw['frame.time_utc'])
        df_raw = df_raw.sort_values(by='time').reset_index(drop=True)
        if df_raw.shape[0] > self.m_max_sample:
            df_raw = df_raw.head(self.m_max_sample)
        #补齐空值
        score = CIoTScore(df_raw,interval=self.m_interval)
        df_raw = score.get_fixed_sample()
        df_raw = score.get_explode_sample(df_raw)
        if df_raw.shape[0] > self.m_max_sample:
            df_raw = df_raw.sample(self.m_max_sample).reset_index(drop=True)
        del df_raw['time']
        df_sample = ioTSample.format(df_raw)
        df_sample = self.select_number_features(df_sample)
        df_sample['noised'] = False
        if df_sample.shape[0] > self.m_train_sample:
            df_sample = df_sample.sample(n=self.m_train_sample).reset_index(drop=True)

        df_noised = ioTSample.add_noise(df_sample,self.m_noise)
        df_noised['noised'] = True
        return pd.concat([df_sample,df_noised],ignore_index=True).reset_index(drop=True)

    def save_packet_label(self,df_sample):
        ioTDbScore = CIoTDbSample("packet_%s"%self.m_attack)
        ioTDbScore.import_data(df_sample)

    def load_packet_label(self):
        ioTDbScore = CIoTDbSample("packet_%s"%self.m_attack)
        return ioTDbScore.load_data()
    
    def get_packet_score(self):
        ioTDbScore = CIoTDbSample("score_%s"%self.m_attack)
        self.df_score = ioTDbScore.load_score()
        measure = self.m_measure
        df_score = self.df_score.copy(deep=True)
        df_score['score'] = df_score[measure]
        max_samples = df_score['group'].nunique() * 5
        df_score = CSHSample.resample_group_balance(df_score,y_column="Label",max_samples=max_samples)
        df_score['Level'] = CIoTPrompt.get_level(df_score,measure)
        df_score['Level_Sum'] = CIoTPrompt.get_level(df_score,'sum')

        #df_score[['id','normal','group','Label','score','sum','Level']]
        all_data = []
        ioTSample = CIoTSample()
        for group,df_tmp in tqdm(df_score.groupby("group"),desc='get packet score'):
            df_tmp = df_tmp.reset_index(drop=True)
            is_normal_list = df_tmp['normal'].tolist()
            id_list = df_tmp['id'].tolist()
            df_sample = ioTSample.get_sample_by_id_list(self.m_attack,is_normal_list,id_list,withFormat=True)
            df_sample = CIoTPrompt.select_number_features(df_sample)
            if df_sample.shape[0] <= 0:
                print("not sample found",is_normal_list,id_list)
            del df_sample['Label']
            df_flattened = CIoTPrompt.get_flat(df_sample)
            df_flattened['group'] = df_tmp.iloc[0]['group']
            df_flattened['Label'] = df_tmp.iloc[0]['Label']
            df_flattened['score'] = df_tmp.iloc[0]['score']
            df_flattened['sum'] = df_tmp.iloc[0]['sum']
            df_flattened['Level'] = df_tmp.iloc[0]['Level']
            df_flattened['Level_Sum'] = df_tmp.iloc[0]['Level_Sum']
            all_data.extend(json.loads(df_flattened.to_json(orient='records')))
            
        df_sample = pd.DataFrame(all_data)
        df_sample = df_sample.fillna(0)
        df_sample = ioTSample.format_flat(df_sample)
        df_sample['noised'] = False
        if df_sample.shape[0] > self.m_train_sample:
            df_sample = df_sample.sample(n=self.m_train_sample).reset_index(drop=True)

        df_noised = ioTSample.add_noise_flat(df_sample,self.m_noise)
        df_noised['noised'] = True
        df_result = pd.concat([df_sample,df_noised],ignore_index=True).reset_index(drop=True)
        return df_result

    def save_packet_score(self,df_sample):
        folder = "%s/packet-score"%(g_sample_root)
        os.makedirs(folder, exist_ok=True)
        file_name = "%s/%s.csv"%(folder,self.m_attack)
        df_sample.to_csv(file_name,index=False)
    
    def load_packet_score(self):
        folder = "%s/packet-score/"%(g_sample_root)
        file_name = "%s/%s.csv"%(folder,self.m_attack)
        return pd.read_csv(file_name)

def create_model_samples(attack,kind): 
    mlSample = IoTMLSample(attack)
    
    if kind in ['all','packet-label']:           #用于ML,创建packet-Label的数据集
        print("create packet-Label",attack)
        df_sample = mlSample.get_packet_label()
        mlSample.save_packet_label(df_sample)

    if kind in ['all','packet-score']:         #用于ML，创建滑动窗口算分，生成的packet-score的数据集
        print("create packet-score",attack)
        df_sample = mlSample.get_packet_score()
        mlSample.save_packet_score(df_sample)

    print("finished")

def main(): 
    parser = argparse.ArgumentParser(description='create attack sample')
    parser.add_argument('--attack', dest='attack', default="all",help='"Port Scanning attack"')
    parser.add_argument('--kind', dest='kind', default="all",help='packet-label|packet-score')
    args = parser.parse_args()
    kind = args.kind
    attack = args.attack
    print('python ./common/IoTMLSample.py --attack="Port Scanning attack"')
    print(attack,kind)

    if attack == "all":
        for attack1 in CIoTSample().get_attack_type():
            create_model_samples(attack1,kind)
    else:
        create_model_samples(attack,kind)

if __name__ == "__main__":
    main()
