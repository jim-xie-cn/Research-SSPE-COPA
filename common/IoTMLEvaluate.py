import os,json,sys,logging
sys.path.append("./share")
sys.path.append("./common")
sys.path.append("./ml")
import pandas as pd
import json
from tqdm import tqdm
from IoTCommon import CIoTCommon
from IoTTotalFeature import CIoTTotalFeature
from IoTSample import CIoTSample
from SHSample import CSHSample
from SHDataProcess import CSHDataProcess
from SHFeatureSelect import CSHFeature
from Config import g_data_root,get_attack_score,g_resolved_columns
from SHDataEDA import CSHDataDistribution,CSHDataTest
from SHModelRegression import CSHModelRegression
from SHModelClassify import CSHModelClassify
from SHEvaluation import CSHEvaluate,CSHSimilarity
from IoTNoise import CIoTNoise
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import h2o
from Config import g_data_root
from datetime import datetime
warnings.simplefilter("ignore")
g_token_root = "%stoken/"%g_data_root
g_feature_root = "%s/features"%g_data_root
g_result_root = "%sresult"%g_data_root
'''
1. 使用COAP和SSPE方法计算频谱
2. 将频谱做为标签，训练模型
3. 对模型输出的频谱进行二值化，将流量划分为正常和攻击流量
4. 分别对比原始的Session、COAP和SSPE方法，提高噪音比例，观察模型的下降
6. Session方式下降最多，COAP和SSPE差别不大
'''
class CClassifyEvaluate:

    def __init__(self,attack):
        self.m_model = CSHModelClassify()
        self.m_col_x = None
        self.m_col_y = None
        
    #根据样本中Label的比例，对score/sum进行二值化
    def get_level(self,attack,df_data,measure):
        ioTPrompt = CIoTPrompt(attack)
        df_label = ioTPrompt.load_packet_label()
        df_label = df_label[df_label['noised']==False].reset_index(drop=True)
        value_ratio = (df_label['Label'].value_counts() ) / len(df_label['Label'])
        df_sample = df_data.copy(deep=True)
        df_sample['score'] = df_sample[measure]
        bins,intervals = CIoTCommon.get_level_by_score(df_sample,value_ratio)
        return bins.astype(int)
        
    def load_sample(self,attack,col_y):
        ioTPrompt = CIoTPrompt(attack)
        if col_y == 'Label':
            df_total = ioTPrompt.load_packet_label()
            df_total = df_total[df_total['noised']==False].reset_index(drop=True)
        else:
            df_total = ioTPrompt.load_packet_score()
            df_total = df_total[df_total['noised']==False].reset_index(drop=True)
            df_total['Level'] = self.get_level(attack,df_total,'score')
            df_total['Level_Sum'] = self.get_level(attack,df_total,'sum')
        
        df_total = CIoTModel(attack).select_packet_features(df_total)
        try:
            df_total = CSHSample.resample_smote(df_total,y_column=col_y).reset_index(drop=True)
            print(col_y,df_total[col_y].value_counts())
        except:
            return pd.DataFrame(),pd.DataFrame()

        df_base = df_total.drop(['id','time','protocol','noised','group','score','sum'], axis=1,errors='ignore')
        ds_label = df_base[col_y]
        df_base = df_base.drop(['Level_Sum','Level','Label'], axis=1,errors='ignore')
        df_base[col_y] = ds_label
        self.m_col_y = col_y
        self.m_col_x = df_base.keys().tolist()
        for col in self.m_col_x:
            if col.find("FLAT") < 0 and col != col_y and col_y != "Label":
                print("Sample includes error column ",col) 
        self.m_col_x.remove(col_y)
        df_sample,scale_colums = CSHDataProcess.get_scale(df_base,x_columns=self.m_col_x,y_column=self.m_col_y)
        df_train,df_test = CSHSample.split_dataset(df_sample)
        return df_train,df_test
  
    def train(self,df_train):
        self.m_model.train(df_train,x_columns=self.m_col_x,y_column=self.m_col_y,train_ratio = 0)
    
    def add_noise(self,df_test,noised_columns = [],ratio=0.1):
        df_data = df_test.copy(deep = True)
        dtypes_dict = df_test.dtypes.to_dict()
        for feature in df_data.keys().tolist():
            if feature in [self.m_col_y]:
                continue
            if not (feature in noised_columns):
                continue
            type1 = dtypes_dict[feature]
            df_data[feature] = CIoTNoise.add_noise(feature,type1,df_data,ratio)
        return df_data
        
    def test(self,df_test,noised_columns=[],ratio=0.1):
        if noised_columns == []:
            noised_columns = self.m_col_x
        if ratio == 0:
            df_noised = df_test
        else:
            df_noised = self.add_noise(df_test,noised_columns,ratio=ratio)
        df_result = self.m_model.evaluate(df_noised,x_columns=self.m_col_x,y_column=self.m_col_y)
        df_result['ratio'] = ratio
        return df_result
'''
1. 使用COAP和SSPE方法计算频谱
2. 使用频谱做为标签，训练模型
3. 使用模型生成预测的频谱
4. 对比不同攻击的攻击频谱的相似度，识别攻击类型
5. 使用SSPE训练的模型，生成的频谱更稳定，可以有效识别攻击的类型
'''
class CRegressionEvaluate:

    def __init__(self,attack):
        self.m_model = CSHModelRegression()
        self.m_col_x = None
        self.m_col_y = None
        
    def load_sample(self,attack,col_y):
        ioTPrompt = CIoTPrompt(attack)
        df_total = ioTPrompt.load_packet_score()
        df_total = CIoTModel(attack).select_packet_features(df_total)
        df_total = df_total[df_total['noised']==False].reset_index(drop=True)
        df_base = df_total.drop(['id','time','protocol','Level','Level_Sum','noised','group','Label'], axis=1,errors='ignore')
        if col_y == 'score':
            del df_base['sum']
        else:
            del df_base['score']

        self.m_col_y = col_y
        self.m_col_x = df_base.keys().tolist()
        for col in self.m_col_x:
            if col.find("FLAT") < 0 and col != col_y:
                print("Sample includes error column ",col)
        self.m_col_x.remove(col_y)
        
        df_sample,scale_colums = CSHDataProcess.get_scale(df_base,x_columns=self.m_col_x,y_column=self.m_col_y)
        df_train,df_test = CSHSample.split_dataset(df_sample)
        return df_train,df_test
  
    def train(self,df_train):
        self.m_model.train(df_train,x_columns=self.m_col_x,y_column=self.m_col_y,train_ratio = 0)
    
    def add_noise(self,df_test,noised_columns = [],ratio=0.1):
        df_data = df_test.copy(deep = True)
        dtypes_dict = df_test.dtypes.to_dict()
        for feature in df_data.keys().tolist():
            if feature in [self.m_col_y]:
                continue
            if not (feature in noised_columns):
                continue
            type1 = dtypes_dict[feature]
            df_data[feature] = CIoTNoise.add_noise(feature,type1,df_data,ratio)
        return df_data
        
    def test(self,df_test,noised_columns=[],ratio=0.1):
        if noised_columns == []:
            noised_columns = self.m_col_x
        if ratio == 0:
            df_noised = df_test
        else:
            df_noised = self.add_noise(df_test,noised_columns,ratio=ratio)
        df_pred = self.m_model.predict(df_noised,x_columns=self.m_col_x,y_column=self.m_col_y)
        df_pred['ratio'] = ratio
        return df_pred
        
class CIoTAttack:

    def __init__(self,df_score):
        self.df_score = df_score.copy(deep=True)
        self.df_spectrum = pd.DataFrame()
        for item,df_tmp in self.df_score.groupby(["attack",'model','kind','ratio']):
            attack = item[0]
            if attack in self.df_spectrum:
                continue
            spectrum = df_tmp[df_tmp['ratio'] == 0]['true'].reset_index(drop = True)
            self.df_spectrum[attack] = spectrum
        self.df_spectrum = self.df_spectrum.fillna(0)
        
    def calculate_distance(self,spectrum,n_sample_count = 1):
        sp_data = spectrum.copy(deep=True)
        if len(sp_data) < self.df_spectrum.shape[0]:
            num_missing = self.df_spectrum.shape[0] - len(sp_data)
            s_missing = pd.Series([0] * num_missing)
            sp_data = pd.concat([sp_data, s_missing], ignore_index=True)
        sp_data = sp_data.reset_index(drop=True)       
        
        all_data = []
        for attack in self.df_spectrum.columns:
            for n_sample in range(n_sample_count):
                sp_base = self.df_spectrum[attack].reset_index(drop=True)
                df_data = pd.DataFrame()
                df_data['sp_data'] = sp_data.tolist()
                df_data['sp_base'] = sp_base.tolist()

                if n_sample_count > 1:
                    data_indices = df_data.sample(n=1000).index
                    df_data  = df_data.loc[sorted(data_indices)]
                    df_data = df_data.reset_index(drop=True)
                    
                df_data['sp_data'] =  ( df_data['sp_data'] - df_data['sp_data'].mean() ) / df_data['sp_data'].std()
                df_data['sp_base'] =  ( df_data['sp_base'] - df_data['sp_base'].mean() ) / df_data['sp_base'].std()

                s = CSHSimilarity(df_data['sp_data'].tolist(),df_data['sp_base'].tolist())
                tmp = {}
                tmp['target'] = attack
                tmp['n_sample'] = n_sample
                tmp['Cosine'] = s.Cosine()
                tmp['Pearson'] = s.Pearson()
                tmp['Euclidean'] = s.Euclidean()
                tmp['EMD'] = s.EMD()
                tmp['KS'] = s.KSTest()[1]
                tmp['Manhattan'] = s.Manhattan()
                tmp['Minkowski'] = s.Minkowski()
                tmp['Jaccard'] = s.Jaccard()
                all_data.append(tmp)
                
        return pd.DataFrame(all_data)

    def get_distance(self,n_sample_count = 1):
        df_distance = pd.DataFrame()
        for item,df_tmp in tqdm(self.df_score.groupby(['attack','model','kind','ratio'])):
            df_tmp = self.calculate_distance(df_tmp['predict'],n_sample_count=n_sample_count)
            df_tmp['attack'] = item[0]
            df_tmp['model'] = item[1]
            df_tmp['kind'] = item[2]
            df_tmp['ratio'] = item[3]
            df_distance = pd.concat([df_distance,df_tmp],ignore_index=True)
        return df_distance.reset_index(drop = True) 

    def get_attack(self,df_distance,measure = 'Cosine'):
        all_data = []
        for item,df_tmp in tqdm(df_distance.groupby(['attack','model','kind',"n_sample","ratio"])):
            tmp = {}
            tmp['attack'] = item[0]
            tmp['model'] = item[1]
            tmp['kind'] = item[2]
            tmp['n_sample'] = item[3]
            tmp['ratio'] = item[4]
            df_tmp = df_tmp.reset_index(drop=True)
            if measure in ['Cosine','Pearson','Jaccard','KS']:
                target_index = df_tmp[measure].idxmax()
            else:
                target_index = df_tmp[measure].idxmin()
            tmp['target'] = df_tmp.iloc[target_index]['target']
            all_data.append(tmp)
        return pd.DataFrame(all_data)
         
    def get_accuracy(self,df_attack):
        df_data = df_attack.copy(deep = True)
        df_data['flag'] = (df_data['attack'] == df_data['target'])
        all_data = []
        for item,df_tmp in df_data.groupby(["attack","model","kind","n_sample","ratio"]):
            df_temp = df_tmp[df_tmp['flag']]
            tmp = {}
            tmp['attack'] = item[0]
            tmp['model'] = item[1]
            tmp['kind'] = item[2]
            tmp['n_sample'] = item[3]
            tmp['ratio'] = item[4]
            tmp['accuracy'] = df_temp['flag'].shape[0]/len(df_tmp)
            all_data.append(tmp)
        return pd.DataFrame(all_data)
        
def train_test(attack,measure ):
    if measure in ['sum','score']:
        evaluter = CRegressionEvaluate(attack)
    else:
        evaluter = CClassifyEvaluate(attack)
        
    df_train,df_test = evaluter.load_sample(attack,measure)
    
    if len(df_train) <=0 or len(df_test) <=0:
        print("jim_error train/test is empty",attack,measure)
        return pd.DataFrame()

    if df_test[measure].nunique() <=1:
        print("jim_error test is None",attack,measure)
        return pd.DataFrame()

    if df_train[measure].nunique() <=1:
        print("jim_error train is None",attack,measure)
        return pd.DataFrame()
    
    evaluter.train(df_train)
    df_result = pd.DataFrame()
    
    noised_columns = []
    for col in evaluter.m_col_x:
        if col == measure:
            continue
        if measure == 'Label':
            noised_columns.append(col)
        elif measure in ['Level','Level_Sum']:
            if col.find("FLAT") > 0 and len(noised_columns) <= 46 : #基线session中，有46个特征添加噪音，这里也给46个特征添加噪音
                noised_columns.append(col)                
        elif measure in ['sum','score']:
            noised_columns.append(col)
        else:
            print("Unknown measure",measure)

    for ratio in tqdm(np.linspace(0, 1, 11)):
        df_tmp = evaluter.test(df_test,noised_columns=noised_columns,ratio=ratio)
        df_result = pd.concat([df_result,df_tmp], ignore_index=True)
    df_result['kind'] = measure
    return df_result.reset_index(drop=True)
    
def show_result(df_result,isClassify):
    for model in df_result['model'].unique():
        df_tmp = df_result[df_result['model']==model]
        df_tmp = df_tmp.drop(['model','kind'], axis=1,errors='ignore')
        df_tmp = df_tmp.groupby('ratio').mean()
        df_tmp = df_tmp.sort_values(by='ratio').reset_index()
        if isClassify:
            df_long = pd.melt(df_tmp,id_vars=['ratio'], value_vars=['recall', 'precision','accuracy','f1_score'], var_name='metric', value_name='value')
        else:
            df_long = pd.melt(df_tmp,id_vars=['ratio'], value_vars=['mse', 'rmse','mae','r2'], var_name='metric', value_name='value')
        sns.barplot(x='ratio', y='value',hue='metric',orient='v',ci=None,data=df_long)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.110), ncol=4)
        plt.xlabel('Noise ratio', fontsize=10)
        plt.ylabel('Metric value', fontsize=10)
        plt.title('Performance of %s'%model, fontsize=12)
        plt.show()

def main():
    h2o.connect(verbose=False)
    df_class_result = pd.DataFrame()
    df_regression_result = pd.DataFrame()
    
    for attack in CIoTSample().get_attack_type():
        df_class_label = train_test(attack,'Label')
        if df_class_label.shape[0] > 0:
            df_class_label['attack'] = attack
            df_class_result = pd.concat([df_class_result,df_class_label])

        df_class_score = train_test(attack,'Level')
        if df_class_score.shape[0] > 0:
            df_class_score['attack'] = attack
            df_class_result = pd.concat([df_class_result,df_class_score])

        df_class_sum = train_test(attack,'Level_Sum')
        if df_class_sum.shape[0] > 0:
            df_class_sum['attack'] = attack
            df_class_result = pd.concat([df_class_result,df_class_sum])

        df_regression_sum = train_test(attack,'sum')
        df_regression_sum['attack'] = attack
        df_regression_result = pd.concat([df_regression_result,df_regression_sum])

        df_regression_score = train_test(attack,'score')
        df_regression_score['attack'] = attack
        df_regression_result = pd.concat([df_regression_result,df_regression_score])
        
    df_class_result = df_class_result.reset_index(drop=True)
    df_regression_result = df_regression_result.reset_index(drop=True)

    df_class_result[df_class_result['kind']=='sum'].reset_index(drop=True).to_csv("./IIoTSet/result/二分类-COAP.csv")
    df_class_result[df_class_result['kind']=='score'].reset_index(drop=True).to_csv("./IIoTSet/result/二分类-SSPE.csv")

    df_regression_result.to_csv("./IIoTSet/result/attack-spectrums.csv")
    
    df_score = pd.read_csv("./IIoTSet/result/attack-spectrums.csv",index_col=0)
    df_distance = CIoTAttack(df_score).get_distance(n_sample_count=10)
    df_distance.to_csv("./IIoTSet/result/distance.csv")

if __name__ == "__main__":
    main()
