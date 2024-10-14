import os,json,sys,logging
sys.path.append("./share")
sys.path.append("./common")
import pandas as pd
import json,warnings
from tqdm import tqdm
from IoTCommon import CIoTCommon
from IoTTotalFeature import CIoTTotalFeature
from IoTDatabase import CIoTDbSample
from IoTSample import CIoTSample
from SHSample import CSHSample
from SHDataProcess import CSHDataProcess
from Config import g_data_root,get_attack_score,g_d_modes
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
from datetime import datetime
from sklearn.metrics import jaccard_score
from scipy.stats import skew, kurtosis
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter

warnings.simplefilter("ignore")
np.random.seed(42)
import argparse
g_sample_root = "%ssample"%g_data_root

class CIoTScore:
    
    def __init__(self,df_raw,d_mode = [4,8,64], interval = None):
        self.df_raw = df_raw.copy(deep = True)
        self.d_mode = d_mode
        if interval == None:
            interval = CIoTScore.get_fixed_interval(df_raw)
            print("auto interval ",interval)
        self.m_interval = interval
        if interval != None:
            self.df_raw['time'] = pd.to_datetime(self.df_raw['frame.time_utc'])
            self.df_raw['time'] = self.df_raw['time'].dt.floor(self.m_interval)
            self.df_normal = CIoTSample().get_sensor_sample("Normal",withFormat=False)
            self.df_normal['time'] = pd.to_datetime(self.df_normal['frame.time_utc'])
            self.df_normal['time'] = self.df_normal['time'].dt.floor(self.m_interval)
            for key in self.df_raw.keys().tolist():
                if not key in self.df_normal.keys().tolist():
                    self.df_normal[key] = np.NaN
            to_drop_columns = []
            for key in self.df_normal.keys().tolist():
                if not key in  self.df_raw.keys().tolist():
                    to_drop_columns.append(key)
            self.df_normal = self.df_normal.drop(to_drop_columns, axis=1,errors='ignore')
    
    #将转换后的样本分配类型
    def set_type(self,df_data):
        df_sampe = df_data.copy(deep=True)
        dtypes_dict = self.df_raw.dtypes.to_dict()
        for key in dtypes_dict:
            type = dtypes_dict[key]
            if type in ['int','int64','float']:
                df_sampe[key] = df_sampe[key].fillna(0)
            df_sampe[key] =  df_sampe[key].astype(type)
        return df_sampe
            
    #返回原始样本
    def get_raw_sample(self):
        return self.df_raw

    #返回原始聚合后的样本
    def get_aggregate_sample(self):
        df_data = self.df_raw.copy(deep = True)
        df_data['time'] = pd.to_datetime( df_data['frame.time_utc'])
        df_data['time'] = df_data['time'].dt.floor(self.m_interval)
        df_data['normal'] = False
        max_id = df_data['id'].max()
        return max_id,df_data.groupby('time',as_index=False).agg(list)
    
    #返回爆炸后的样本
    def get_explode_sample(self,df_aggregated):
        groupped_columns = df_aggregated.keys().tolist()
        groupped_columns.remove("time")
        df_exploded = df_aggregated.explode(groupped_columns).reset_index(drop=True)
        return self.set_type(df_exploded)
        
    #返回补齐时间后的样本
    def get_fixed_sample(self):
        max_id,df_data = self.get_aggregate_sample()
        startTime,endTime = df_data['time'].min(),df_data['time'].max()
        date_rng = pd.date_range(start=startTime, end=endTime, freq=self.m_interval)
        df_time = pd.DataFrame({"time":date_rng})
        df_time = df_time[~df_time['time'].isin(df_data['time'].unique().tolist())]
        df_time = df_time.reset_index(drop=True)
        df_tmp =  self.df_normal.sample(n=df_time.shape[0],replace = True).reset_index(drop=True)
        #df_tmp['id'] = df_time.index+max_id+1
        df_tmp['time'] = df_time['time']
        df_tmp['frame.time_utc'] = df_tmp['time']
        df_tmp['normal'] = True
        df_tmp = df_tmp.groupby('time',as_index=False).agg(list)
        df_total = pd.concat([df_data,df_tmp],ignore_index=True).sort_values(by='time')
        return df_total.reset_index(drop=True)
    
    @staticmethod
    def get_fixed_interval(df_raw):
        startTime,endTime = df_raw['time'].min(),df_raw['time'].max()
        deltTime = (endTime-startTime).seconds
        print("Delt time",deltTime)
        if deltTime > 3600 * 24:
            return "30S"
        if deltTime > 3600 * 12:
            return "10S"
        if deltTime > 3600:
            return "1S"
        if deltTime > 300:
            return "100L"
        if deltTime > 60:
            return "10L"
        return None

    def get_time_range(self):
        max_id,df_data = self.get_aggregate_sample()
        startTime,endTime = df_data['time'].min(),df_data['time'].max()
        date_rng = pd.date_range(start=startTime, end=endTime, freq=self.m_interval)
        return startTime,endTime,len(date_rng)

    # 原始样本分值
    def get_raw_score(self,window_size=30,step=1):
        def score_call_back(ioTScore,df_window,d_mode,interval):
            return get_attack_score(df_window['Label'],d_mode),df_window
        return self.get_score(self.df_raw,score_call_back,window_size=window_size,step=step)
        
    # 补全时间后，聚合的分值
    def get_fix_score(self,window_size=30,step=1):
        df_fix = self.get_fixed_sample()
        def score_call_back(ioTScore,df_window,d_mode,interval):
            ds_label = df_window['Label'].apply(lambda x: np.sum(x))
            df_explored = ioTScore.get_explode_sample(df_window)
            return get_attack_score(ds_label,d_mode),df_explored
        return self.get_score(df_fix,score_call_back,window_size=window_size,step=step)

    # 补全时间后，爆炸的分值
    def get_explode_score(self,window_size=30,step=1):
        df_fix = self.get_fixed_sample()
        df_explode = self.get_explode_sample(df_fix)
        def score_call_back(ioTScore,df_window,d_mode,interval):
            return get_attack_score(df_window['Label'],d_mode),df_window
        return self.get_score(df_explode,score_call_back,window_size=window_size,step=step)
    
    def get_score(self,df_data,score_call_back,window_size=30,step=1):
        df_sample = pd.DataFrame()
        if len(df_data) - window_size + step < 0:
            print("data deficiencies")
            return df_data
        for d_mode in self.d_mode:
            for i in tqdm(range(len(df_data) - window_size + step),desc="caculate score"):
                df_tmp = df_data.iloc[i:i+window_size]
                score,df_window = score_call_back(self,df_tmp,d_mode,self.m_interval)
                df_window['score'] = score
                df_window['group'] = i
                df_window['d_mode'] = d_mode
                df_sample = pd.concat([df_sample,df_window],ignore_index=True)
        df_sample = df_sample.reset_index(drop=True)
        return df_sample
        
    def get_multiple_score(self,window_size=60,step=1):
        
        def get_window_score(ioTScore,df_window):
            ret = {}
            ds_label = df_window['Label'].apply(lambda x: np.sum(x))
            ret['sum'] = ds_label.sum()
            for d_mode in ioTScore.d_mode:
                ret['merge_%d'%d_mode] = get_attack_score(ds_label,d_mode)

            df_exploded = ioTScore.get_explode_sample(df_window)
            ds_label = df_exploded['Label']
            for d_mode in ioTScore.d_mode:
                ret['explod_%d'%d_mode] = get_attack_score(ds_label,d_mode)
            return ret
        
        df_fix_sample = self.get_fixed_sample()
        all_data = []
        for i in tqdm(range(len(df_fix_sample) - window_size + step),desc="caculate score"):
            df_tmp = df_fix_sample.iloc[i:i+window_size]
            score = get_window_score(self,df_tmp)
            df_tmp = self.get_explode_sample(df_tmp)
            df_tmp = df_tmp[['time','id','normal','Label']]
            df_tmp['group'] = i
            df_tmp['interval'] = self.m_interval
            df_tmp['window_size'] = window_size
            df_tmp['step'] = step
            for key in score.keys():
                df_tmp[key] = score[key]
            all_data.extend(json.loads(df_tmp.to_json(orient='records')))
        return pd.DataFrame(all_data)

class CIoTScoreTest:
    def __init__(self,attack):
        self.m_attack = attack
        self.df_score = CIoTDbSample("score_%s"%self.m_attack).load_score()
        name_map = {
            "sum":"A.sum",
            "merge_2":"B.merge_2",
            "merge_4":"C.merge_4",
            "merge_8":"D.merge_8",
            "merge_16":"E.merge_16",
            "merge_32":"F.merge_32",
            "merge_64":"G.merge_64",
            "merge_128":"H.merge_128",
            "merge_256":"J.merge_256",
            "explod_2":"K.explod_2",
            "explod_4":"L.explod_4",
            "explod_8":"M.explod_8",
            "explod_16":"N.explod_16",
            "explod_32":"O.explod_32",
            "explod_64":"P.explod_64",
            "explod_128":"Q.explod_128",
            "explod_256":"R.explod_256"
        }
        self.m_measure_list = []
        for key in name_map:
            new_name = name_map[key]
            self.df_score[new_name] = self.df_score[key]
            del self.df_score[key]
            self.m_measure_list.append(new_name)
        self.m_measure_list.sort()
    
    def get_measures(self):
        return self.m_measure_list

    def get_score(self):
        return self.df_score
        
    @staticmethod
    def get_jaccard(df_data,measure):
        A = np.array(df_data['Label'].tolist())
        B = np.array(df_data['Level'].tolist())
        return jaccard_score(A, B)
        
    @staticmethod
    def get_distance(df_data,measure):
        df_tmp = df_data.groupby("group").sum().reset_index()
        df_tmp[measure] = (df_tmp[measure] - df_tmp[measure].mean()) / df_tmp[measure].std() #- df_tmp['score'].min())
        df_tmp['Label'] = (df_tmp['Label'] - df_tmp['Label'].mean()) / df_tmp['Label'].std() #- df_tmp['score'].min())
        df_tmp = df_tmp.sort_values(by='group')
        def get_euclidean_distance(vec1, vec2):
            diff = vec1 - vec2
            return np.sqrt(np.sum(diff ** 2))
        return get_euclidean_distance(df_tmp['Label'],df_tmp[measure])

    @staticmethod
    def get_level(df_data,measure):
        df_sample = df_data.copy(deep=True)
        value_ratio = (df_sample['Label'].value_counts() ) / len(df_sample['Label'])
        df_sample['score'] = df_sample[measure]
        bins,intervals = CIoTCommon.get_level_by_score(df_sample,value_ratio)
        return bins.astype(int)
        
    @staticmethod
    def get_skew_kurt(df_data,measure):
        ds_tmp = (df_data[measure] - df_data[measure].mean()) / df_data[measure].std()
        return skew(ds_tmp),kurtosis(ds_tmp, fisher=True)
        
    def get_result(self):
        df_data = self.df_score.copy(deep=True)
        tmp = {}
        tmp['attack'] = self.m_attack
        all_data = []    
        for measure in self.get_measures():
            tmp['kind'] = measure
            df_data['Level'] = CIoTScoreTest.get_level(df_data,measure)
            tmp['distance'] = CIoTScoreTest.get_distance(df_data,measure)
            tmp['jaccard'] = CIoTScoreTest.get_jaccard(df_data,measure)
            tmp['skew'],tmp['kurt'] = CIoTScoreTest.get_skew_kurt(df_data,measure)
            #numeric_df = df_data.select_dtypes(include=['number'])
            df_tmp1 = df_data.groupby("group").sum().reset_index().sort_values(by='group').reset_index(drop=True)
            df_tmp2 = df_data[['group',measure]].groupby("group").mean().reset_index().sort_values(by='group').reset_index(drop=True)
            df_tmp = df_tmp1
            df_tmp[measure] = df_tmp2[measure]
            for i,row in df_tmp.iterrows():
                tmp['group'] = row['group']
                tmp['Label'] = row['Label']
                tmp['Level'] = row['Level']
                tmp['score'] = row[measure]
                all_data.append(tmp.copy())
                
        return pd.DataFrame(all_data)
    
    def show_result(self,df_result,n_samples= None,nbins=100,min_score = None):
        value_ratio = (self.df_score['Label'].value_counts() ) / len(self.df_score['Label'])        
        df_data = df_result.sort_values(by='distance').reset_index(drop=True)

        measure_count = len(self.get_measures())
        
        fig, axes = plt.subplots(measure_count, 2, figsize=(16, 64))
        
        ax_off = 0
        
        for kind,df_tmp in df_data.groupby("kind"):
            if n_samples != None:
                df_tmp = df_tmp.sample(n_samples,replace=True)
            if min_score != None:
                df_tmp = df_tmp[df_tmp['score'] > min_score]
            df_tmp = df_tmp.sort_values(by='group').reset_index(drop=True)
            temp = json.loads(df_tmp.iloc[0].to_json())
            
            tmp = {}
            tmp['kind'] = kind.split(".")[1]
            tmp['ratio']="%4.2f"%value_ratio.loc[0]
            tmp['distance']="%4.2f"%temp['distance']
            tmp['jaccard']="%4.2f"%temp['jaccard']
            tmp['skew'] = "%4.2f"%temp['skew']
            tmp['kurt'] = "%4.2f"%temp['kurt']
            ds_tmp = self.get_score()[kind]

            if min_score != None:
                ds_tmp = ds_tmp[ds_tmp > min_score]
            
            #ds_tmp = (ds_tmp - ds_tmp.mean())/ds_tmp.std()

            ds_tmp.hist(bins=nbins,ax=axes[ax_off,0])
            
            df_tmp['score'] = (df_tmp['score'] - df_tmp['score'].mean()) / df_tmp['score'].std()
            df_tmp['Level'] = (df_tmp['Level'] - df_tmp['Level'].mean()) / df_tmp['Level'].std()

            if kind.find("sum") > 0 :
                df_tmp['Attack Count'] = (df_tmp['Label'] - df_tmp['Label'].mean()) / df_tmp['Label'].std()
                df_tmp['COAP'] = df_tmp['score']
                #df_tmp[['Attack Count','COAP']].plot(ax=axes[ax_off,1])
                df_tmp['Attack Count'].plot(ax=axes[ax_off, 1],label="")  
                df_tmp['COAP'].plot(ax=axes[ax_off, 1],label='COAP') 
                tmp1 = "Histogram of COAP"
                axes[ax_off,0].set_title(tmp1)
            else:
                df_tmp['COAP'] = (df_tmp['Label'] - df_tmp['Label'].mean()) / df_tmp['Label'].std()
                df_tmp['SSPE'] = df_tmp['score']
                df_tmp[['SSPE','COAP']].plot(ax=axes[ax_off,1])
                tmp1 = "Histogram of SSPE"
                axes[ax_off,0].set_title(tmp1)
            
            #print(attack,tmp)
            axes[ax_off,1].set_title(json.dumps(tmp)+"\n\n")
            axes[ax_off, 1].legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=1)

            axes[ax_off,0].yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
            axes[ax_off,0].ticklabel_format(style='plain', axis='y')
            axes[ax_off,0].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x):,}'))

            ax_off += 1
        
        plt.tight_layout()
        plt.savefig("./pic/%s.png"%(self.m_attack.replace(" ","_")))
        plt.show()
        
def Create_Normal_Sample():
    test = CIoTSample()
    df_normal = pd.DataFrame()
    for trafficType in tqdm(test.get_sensor_type(),desc="Importing Normal"):
        df_raw = test.get_sensor_sample(trafficType,withFormat=False,pageID=1)
        df_normal = pd.concat([df_normal,df_raw],ignore_index=True)
    df_normal = df_normal.reset_index(drop = True)
    dbSample = CIoTDbSample("Normal")
    dbSample.create_table()
    dbSample.clear_sample()
    dbSample.import_sample(df_normal)

def create_attack_score(attack):
    from IoTMLSample import g_sample_cfg

    d_mode = g_d_modes
    interval = g_sample_cfg[attack]['interval']
    window_size = g_sample_cfg[attack]['window_size']    
    max_sample = g_sample_cfg[attack]['max_sample']

    df_raw = CIoTSample().get_attack_sample(attack,withFormat=False,filtered=True)
    df_raw.loc[df_raw['Label']==-1,"Label"]=0

    #防止数据量过大
    df_raw['time'] = pd.to_datetime(df_raw['frame.time_utc'])
    #df_raw = CSHSample.resample_balance(df_raw,max_samples=max_sample)

    df_raw = df_raw.sort_values(by='time').reset_index(drop=True)

    if df_raw.shape[0] > max_sample:
        df_raw = df_raw.head(max_sample)
    
    if df_raw.shape[0] <= 0:
        print("No enough data")
        return

    ioTScore = CIoTScore(df_raw,d_mode,interval=interval)
    if ioTScore.m_interval == None:
        print("No enough data")
        return

    df_sample = ioTScore.get_multiple_score(window_size = window_size)
       
    if df_sample.shape[0] > 0:
        dbSample = CIoTDbSample("score_%s"%attack)
        dbSample.import_score(df_sample)
    
    print("create score finished")

def Create_Score_Sample(attack):
    if attack == None or attack == 'all':
        print("create all attack score") 
        for attack1 in CIoTSample().get_attack_type():
            print(attack1)
            create_attack_score(attack1)
    else:
        print("create attack score",attack)
        create_attack_score(attack)

def test_attack_score(attack = 'DDoS HTTP Flood Attacks'):
    ioTScore = CIoTScoreTest(attack)
    df_result = ioTScore.get_result()
    ioTScore.show_result(df_result,n_samples=None)
    print(df_result)

def main():
    parser = argparse.ArgumentParser(description='calculate attack score')
    parser.add_argument('--action', dest='action', default="",help='create_normal|create_score|get_score')
    parser.add_argument('--attack', dest='attack', default="all",help='"Port Scanning attack"')
    args = parser.parse_args()
    action = args.action
    attack = args.attack
    print(action,attack)
    if not action in ['create_normal','create_score','get_score']:
        print('python ./common/IoTScore.py --action=create_normal|create_score|get_score --attack="Port Scanning attack"')
        return

    if action == 'create_normal':
        Create_Normal_Sample()
    elif action == 'get_score':
        test_attack_score(attack)
    elif action == 'create_score':
        Create_Score_Sample(attack)
if __name__ == "__main__":
    main()
