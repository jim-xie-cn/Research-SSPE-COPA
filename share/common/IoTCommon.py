import pandas as pd
import json
import os,json,logging
from Config import g_data_root
from tqdm.notebook import tqdm


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
            temp_folder = "%sdataset/Attack"%(g_data_root)
        else:
            temp_folder = "%sdataset/Normal"%(g_data_root)
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
            temp_folder = "%sdataset/Attack/%s"%(g_data_root,trafficType)
        else:
            temp_folder = "%sdataset/Normal/%s"%(g_data_root,trafficType)        
        df_sample = pd.DataFrame()
        for fi in tqdm(CIoTCommon.get_json_files(temp_folder),desc="loading %s"%trafficType):
            fi = fi.replace("\\","/")
            df_tmp = pd.read_json(fi)
            df_sample = pd.concat([df_sample,df_tmp],ignore_index=True)
        df_sample = df_sample.reset_index(drop=True)
        return df_sample
    
def main():
    trafficType = CIoTCommon.get_category(True)
    for t in trafficType:
        df_sample = CIoTCommon.get_sample(True,t)
        print(df_sample)
        print(df_sample.shape)
        break
    
if __name__ == "__main__":
    main()
