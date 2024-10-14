import pandas as pd
import json
import numpy as np
import os,json,logging
from Config import g_data_root
from tqdm import tqdm

class CIoTCommon:

    @staticmethod
    def get_json_files(directory):
        ret = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.json'):
                    file_name = "%s/%s" % (root, file)
                    ret.append(file_name)
        return ret

    @staticmethod
    def get_csv_files(directory):
        ret = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    file_name = "%s/%s" % (root, file)
                    ret.append(file_name)
        return ret
    
    @staticmethod
    def get_excel_files(directory):
        ret = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.xlsx'):
                    file_name = "%s/%s" % (root, file)
                    ret.append(file_name)
        return ret
    @staticmethod
    def traverse_json(data,result):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    CIoTCommon.traverse_json(value,result)
                else:
                    result[key] = value
        elif isinstance(data, list):
            for index, item in enumerate(data):
                CIoTCommon.traverse_json(item,result)

    @staticmethod
    def get_total_features(file_name):
        with open(file_name,'r') as fp:
            data = []
            total_columns = {}
            for item in json.loads(fp.read() ):
                temp = {}
                CIoTCommon.traverse_json(item['_source'],temp)
                mainKey = temp['frame.protocols']
                if not mainKey in total_columns:
                    total_columns[mainKey] = []
                for key in temp.keys():
                    if key in total_columns[mainKey]:
                        continue
                    else:
                        tmp = {}
                        tmp['frame.protocols'] = temp['frame.protocols']
                        tmp['feature'] = key
                        total_columns[mainKey].append(key)
                        data.append(tmp.copy())
        return data

    @staticmethod
    def get_feature_data(file_name):
        with open(file_name,'r') as fp:
            data = []
            for item in json.loads(fp.read() ):
                temp = {}
                CIoTCommon.traverse_json(item['_source'],temp)
                data.append(temp)
        return data 
    
    @staticmethod
    def get_category(isAttack = True):
        if isAttack:
            temp_folder = "%ssample/Attack"%(g_data_root)
        else:
            temp_folder = "%ssample/Normal"%(g_data_root)
        ret = []
        for fi in CIoTCommon.get_json_files(temp_folder):
            fi = fi.replace("\\","/")
            trafficType = fi.split("/")[-2]
            if not trafficType in ret:
                ret.append(trafficType)
        ret.sort()
        return ret
            
    @staticmethod
    def get_sample(isAttack,trafficType):
        if isAttack:
            temp_folder = "%ssample/Attack/%s"%(g_data_root,trafficType)
        else:
            temp_folder = "%ssample/Normal/%s"%(g_data_root,trafficType)  
        print(temp_folder)
        df_sample = pd.DataFrame()
        for fi in tqdm(CIoTCommon.get_json_files(temp_folder),desc="loading %s"%trafficType):
            fi = fi.replace("\\","/")
            df_tmp = pd.read_json(fi)
            df_sample = pd.concat([df_sample,df_tmp],ignore_index=True)
        df_sample = df_sample.reset_index(drop=True)
        return df_sample
    
    @staticmethod
    def has_decimal(numbers):
        for num in numbers:
            if num is None:
                continue
            try:
                int_num = int(num)
                if int_num != num:
                    return True
            except:
                continue

        return False
    
    @staticmethod
    def is_number(ds_data):
        ds_tmp = ds_data.fillna(0)
        numeric_series = pd.to_numeric(ds_tmp, errors='coerce')
        non_numeric_elements = ds_tmp[pd.isna(numeric_series)]
        return non_numeric_elements.count() == 0
    
    @staticmethod
    def get_level_by_score( df_data, value_ratio ):
        quantiles = value_ratio.head(value_ratio.shape[0]-1).tolist()
        quantile_edges = np.quantile(df_data['score'], quantiles)
        quantile_edges = np.unique(quantile_edges, return_counts=False)
        bins = [-np.inf] + list(quantile_edges) + [np.inf]
        labels = []
        for i in range(len(bins)-1):
            labels.append(i)
        bins,bin_edges = pd.cut(df_data['score'], bins=bins,labels=labels,include_lowest=True,retbins=True)
        return bins,bin_edges
    
def main():
    trafficType = CIoTCommon.get_category(True)
    for t in trafficType:
        df_sample = CIoTCommon.get_sample(True,t)
        print(df_sample)
        print(df_sample.shape)
        break
    
if __name__ == "__main__":
    main()
