import os,json,sys,logging
from tqdm.notebook import tqdm
import numpy as np
sys.path.append("./share")
sys.path.append("./common")
from SHDataProcess import CSHDataProcess

g_data_root = "./IIoTSet/"

g_sample_cfg = {
    "XSS attacks":{"measure":"explod_256","interval":"10L","noise":0.5,"window_size":60,"max_sample":10000*5},  
    "Password attacks":{"measure":"explod_256","interval":"100L","noise":0.5,"window_size":30,"max_sample":10000*5},  
    "Uploading attack":{"measure":"explod_8","interval":"10L","noise":0.5,"window_size":60,"max_sample":10000*5},  
    "DDoS UDP Flood Attacks":{"measure":"explod_256","interval":"1L","noise":0.5,"window_size":30,"max_sample":10000*5},  
    "DDoS ICMP Flood Attacks":{"measure":"explod_8","interval":"1L","noise":0.5,"window_size":30,"max_sample":10000*5},
    "DDoS TCP SYN Flood Attacks":{"measure":"explod_64","interval":"1L","noise":0.5,"window_size":60,"max_sample":10000*5},
    "DDoS HTTP Flood Attacks":{"measure":"explod_16","interval":"1L","noise":0.5,"window_size":60,"max_sample":10000*5},
    "Backdoor_attack":{"measure":"explod_256","interval":"100L","noise":0.5,"window_size":60,"max_sample":10000*5},
    "MITM (ARP spoofing + DNS) Attack":{"measure":"explod_128","interval":"10L","noise":0.5,"window_size":60,"max_sample":10000*5},
    "SQL injection attack":{"measure":"explod_128","interval":"100L","noise":0.5,"window_size":60,"max_sample":10000*5},
    "OS Fingerprinting attack":{"measure":"explod_64","interval":"10L","noise":0.5,"window_size":60,"max_sample":10000*5},
    "Port Scanning attack":{"measure":"explod_128","interval":"10L","noise":0.5,"window_size":60,"max_sample":10000*5},
    "Vulnerability scanner attack":{"measure":"explod_8","interval":"100L","noise":0.5,"window_size":30,"max_sample":10000*5},
    "Ransomware attack":{"measure":"explod_128","interval":"10L","noise":0.5,"window_size":30,"max_sample":10000*5},
}

g_resolved_columns = ['id','group','protocol','Label','score','sum','Level','Level_Sum','time','frame.time_utc','noised','normal']
g_d_modes = [2,4,8,16,32,64,128,256]

def get_attack_score(x,d,bar=None):
    d_model = d
    seq_length = len(x)
    position_enc = CSHDataProcess.get_transformer_position_encoding(seq_length, d_model)
    position_array = np.sum(position_enc, axis=1)
    label_array = np.array(x).reshape(-1,)
    score_array =  label_array * position_array 
    score = np.sum(score_array)
    if bar: 
        bar.update(1)
    return score

def get_series_score(ds_data,d_model=2,window_size = 120 ):
    df_tmp = ds_data
    bar = tqdm(total=df_tmp.shape[0])
    ds_test = df_tmp.rolling(window=window_size).apply(get_attack_score, raw=True,args=(d_model,bar)).dropna()
    ds_test = ds_test.reset_index(drop=True)
    return ds_test
