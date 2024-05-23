import pandas as pd
import json

class CIoTProtoCategory:
    
    def __init__(self,df_total_features):
        self.m_df_features = df_total_features
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
        for item in self.m_df_features['protocol'].unique().tolist():
            if item.startswith(proto) and item != proto:
                ret.append(item)
        return ret
        
    def __get_feature(self,proto):
        return self.m_df_features[self.m_df_features['protocol'] == proto]['feature'].unique().tolist()
    
    def get_intersection_features(self,proto):
        child_features = set(self.__get_feature(proto))
        for child_proto in self.get_child_protocols(proto):
            tmp_feature = self.m_df_features[self.m_df_features['protocol'].str.startswith(child_proto)]['feature'].unique().tolist()
            if not tmp_feature:
                continue
            if not child_features:
                child_features = set(tmp_feature)
            child_features = child_features & set(tmp_feature)
        return child_features

    def parse(self):
        self.m_all_features = {}
        for proto in self.m_df_features['protocol'].unique():
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
        #root
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
        for proto in self.m_df_features['protocol'].unique():
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
    df_total_raw_features = pd.read_csv("./Edge-IIoTset dataset/features/total_raw_features.csv",index_col=0)
    df_tmp = df_total_raw_features.copy()
    df_tmp = df_tmp[~df_tmp["frame.protocols"].isin(["eth:ethertype:ip:tcp:http:data-text-lines","eth:ethertype:ip:tcp:tls:x509sat:x509sat:x509sat:x509af:x509sat:x509ce:x509ce:x509ce:x509ce:x509sat:x509sat:x509sat:x509sat:x509ce:x509ce:x509sat:x509sat:x509ce:x509ce:tls"])]
    df_tmp = df_tmp[~df_tmp['frame.protocols'].str.contains('ipv6')]
    df_tmp = df_tmp[df_tmp['feature']!="\\r\\n"]

    test = CIoTProtoCategory(df_tmp)

    print(json.dumps(test.get_protocols(),indent=4))

    print(json.dumps(test.get_features("eth"),indent=4))

    print(json.dumps(test.get_features("eth:ethertype"),indent=4))

    print(json.dumps(test.get_features("eth:ethertype:ip"),indent=4))
    
if __name__ == "__main__":
    main()
