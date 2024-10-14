import os,json,sys,logging,random
sys.path.append("../")
sys.path.append("../common")
sys.path.append("../share")
import numpy as np
import pandas as pd
np.random.seed(42)

class CIoTNoise:
    
    @staticmethod
    def get_candidate_index(ds_data,ratio = 0.1):
        num_samples = int(ratio * len(ds_data))
        sample_indices = np.random.choice(ds_data.index, num_samples, replace=False)
        return sample_indices

    @staticmethod
    def add_bool_noise(feature,ds_data,candidate_index):
        ds_tmp = ds_data.astype('bool')
        ds_inverse = ds_tmp ^ 1
        ds_data.iloc[candidate_index] = ds_inverse[candidate_index]
        #print("Add Noise",feature,"noised count",len(candidate_index))
        return ds_data

    @staticmethod
    def add_float_noise(feature,ds_data,candidate_index):
        ds_tmp = ds_data.iloc[candidate_index]
        mean = ds_tmp.mean()
        std_dev = ds_tmp.std()
        noise = np.random.normal(mean, std_dev, len(candidate_index))
        ds_data.loc[candidate_index] += noise
        #print("Add Noise",feature,mean,std_dev,"noised count",len(candidate_index))
        return ds_data

    @staticmethod
    def add_int_noise(feature,ds_data,candidate_index):
        ds_tmp = ds_data.iloc[candidate_index]
        mean = ds_tmp.mean()
        std_dev = ds_tmp.std()
        noise = np.random.normal(mean, std_dev, len(candidate_index)).astype(int)
        ds_data.loc[candidate_index] += abs(noise)
        #print("Add Noise",feature,mean,std_dev,"noised count",len(candidate_index))
        return ds_data

    @staticmethod
    def add_category_noise(feature,ds_data,candidate_index):
        return ds_data

    @staticmethod
    def add_datetime_noise(feature,ds_data,candidate_index):
        return ds_data

    @staticmethod
    def add_string_noise(feature,ds_data,candidate_index):
        return ds_data

    @staticmethod
    def add_noise(feature,type,df_input,ratio):
        ds_data = df_input[feature].copy(deep=True)
        candidate_index = CIoTNoise.get_candidate_index(ds_data,ratio)
        if len(candidate_index) <= 0:
            return ds_data
        if type in ['int','int64']:
            return CIoTNoise.add_int_noise(feature,ds_data,candidate_index)
        elif type == 'string':
            return CIoTNoise.add_string_noise(feature,ds_data,candidate_index)
        elif type == 'binary':
            return CIoTNoise.add_string_noise(feature,ds_data,candidate_index)
        elif type == 'float':
            return CIoTNoise.add_float_noise(feature,ds_data,candidate_index)
        elif type == 'category':
            return CIoTNoise.add_category_noise(feature,ds_data,candidate_index)
        elif type == 'datetime':
            return CIoTNoise.add_datetime_noise(feature,ds_data,candidate_index)
        elif type == 'bool':
            return CIoTNoise.add_bool_noise(feature,ds_data,candidate_index)
        else:
            return ds_data

def main():
    from IoTSample import CIoTSample
    test = CIoTSample()
    trafficType = "Backdoor_attack"
    protocol = "eth:ethertype:ip"
    df_sample = test.get_attack_sample(trafficType=trafficType,protocol=protocol).head(100)
    print(df_sample['tcp.completeness.syn'].unique())
    feature = 'tcp.completeness.syn'
    ds_noised = CIoTNoise.add_noise(feature,'bool',df_sample,0.2)
    print(df_sample[feature])
    print(ds_noised)        
    
if __name__ == "__main__":
    main()
