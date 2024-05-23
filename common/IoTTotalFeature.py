'''
协议对应的特征
'''
import pandas as pd
import json
from Config import g_data_root

class CIoTTotalFeature:
    
    #@staticmethod
    def get_expand_protocols(protocol):
        tmp = protocol.split(":")
        ret = []
        parent=""
        for i in range(len(tmp)):
            parent = tmp[i] if not parent else "%s:%s"%(parent,tmp[i])
            ret.append(parent)
        return ret
        
    #@staticmethod
    def get_expand_features(df_features , totalFeature):
        all_data = []
        for item in json.loads(df_features.to_json(orient='records')):
            protocol = item['protocol']
            for proto in CIoTTotalFeature.get_expand_protocols(protocol):
                for feature in totalFeature.get_features(proto):
                    if feature == item['feature']:
                        tmp = item.copy()
                        tmp['protocol'] = proto
                        all_data.append(tmp)
        return pd.DataFrame(all_data)
    
    def __init__(self):
        df_base_features = pd.read_csv("%s/features/total_base_features.csv"%g_data_root,index_col=0)
        df_valid_features = pd.read_excel("%s/features/features.xlsx"%g_data_root)
        df_valid_features = df_valid_features[df_valid_features['effective'] !="N"]
        df_valid_features = df_valid_features[df_valid_features['count'] > 1].reset_index(drop=True)
        self.m_df_base_features =  df_base_features[df_base_features['feature'].isin(df_valid_features['feature'].unique().tolist())]
        self.m_all_features = {}
        self.parse()
        
    def get_parent_protocol(self,proto):
        def traverse_protocol(proto,total_proto):
            tmp = proto.split(":")
            parent = ":".join(tmp[0:-1])
            if parent:
                total_proto.insert(0,parent)
                return traverse_protocol(parent,total_proto)
            else:
                return total_proto
        total_proto=[]
        return traverse_protocol(proto,total_proto)
    
    def get_child_protocols(self,proto):
        ret = []
        for item in self.m_df_base_features['protocol'].unique().tolist():
            if item.startswith(proto) and item != proto:
                ret.append(item)
        return ret
        
    def __get_feature(self,proto):
        return self.m_df_base_features[self.m_df_base_features['protocol'] == proto]['feature'].unique().tolist()
    
    def get_intersection_features(self,proto):
        child_features = set(self.__get_feature(proto))
        for child_proto in self.get_child_protocols(proto):
            tmp_feature = self.m_df_base_features[self.m_df_base_features['protocol'].str.startswith(child_proto)]['feature'].unique().tolist()
            if not tmp_feature:
                continue
            if not child_features:
                child_features = set(tmp_feature)
            child_features = child_features & set(tmp_feature)
        return child_features

    def parse(self):
        self.m_all_features = {}
        for proto in self.m_df_base_features['protocol'].unique():
            current_proto = ""
            for item in proto.split(":"):
                if current_proto == "":
                    current_proto = item
                else:
                    current_proto = current_proto+":"+item
                if not current_proto in self.m_all_features :
                    self.m_all_features[current_proto] = []
                for feature in self.get_intersection_features(current_proto):
                    if not feature in self.m_all_features[current_proto]:
                        self.m_all_features[current_proto].append(feature)
                        
    def get_features(self,protocol):
        if protocol.lower() in ['eth','eth:ethertype']:
            return self.m_all_features[protocol]

        parent_features = set([])
        for proto in self.get_parent_protocol(protocol):
            feature = self.m_all_features[proto]
            parent_features = parent_features | set(feature)

        current_features =[] 
        for feature in self.m_all_features[protocol]:
            if not feature in list(parent_features):
                current_features.append(feature)
        return current_features
        
    def get_protocols(self):
        ret = []
        for proto in self.m_df_base_features['protocol'].unique():
            current_proto = ""
            for item in proto.split(":"):
                if current_proto == "":
                    current_proto = item
                else:
                    current_proto = current_proto+":"+item
                if not current_proto in ret:
                    ret.append(current_proto)
        return ret

def main():
    test = CIoTTotalFeature()
    print(json.dumps(test.get_protocols(),indent=4))
    print(json.dumps(test.get_features("eth"),indent=4))
    print(json.dumps(test.get_features("eth:ethertype"),indent=4))
    print(json.dumps(test.get_features("eth:ethertype:ip"),indent=4))
    print(json.dumps(test.get_features("eth:ethertype:ip:tcp"),indent=4))

if __name__ == "__main__":
    main()
