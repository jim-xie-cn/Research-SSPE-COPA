# Source code for paper and dataset 

paper:
dataset: https://huggingface.co/datasets/JimXie/IIoTset 

# Development environment prepare
1. setup python packages of ./install/requirements.txt
2. setup mysql，grant root access permission（user name is root，password is iot_admin)
# Data prepare
1. download dataset：

   https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications

   https://huggingface.co/datasets/JimXie/IIoTset/blob/main/features.tar.gz

   https://huggingface.co/datasets/JimXie/IIoTset/blob/main/mysql.tar.gz

   https://huggingface.co/datasets/JimXie/IIoTset/blob/main/sample.tar.gz

   https://huggingface.co/datasets/JimXie/IIoTset/blob/main/dataset.tar.gz

   https://huggingface.co/datasets/JimXie/IIoTset/blob/main/pcap-json.tar.gz

2. uncompress dataset to IIoTset folder
   
3. after uncompressing, the struct of IIoTset folder should like below:

   IIoTset
   
       |-raw

          |-Attack traffic

          |-Normal traffic

          |-Selected dataset for ML and DL

       |-features
          
          |-features.xlsx

          |-sample-features.xlsx

          |-total_base_features.csv
   
       |-dataset

          |-Attack

          |-Normal 

       |-pcap-json
   
          |-Attack

          |-Normal 

       |-sample

          |-packet-score

       |-result
          |-attack-importance.csv
   
          |-binary-classification-COAP.csv
   
          |-attack-performance.csv
                                          
          |-binary-classification-SSPE.csv
   
          |-attack-spectrums.csv
                                    
          |-distance.csv
   
          |-binary-classification-4-features.csv
                             
          |-binary-classification-46-features-without-normalization.csv
      
          |-spectrums-distribution.csv
   
          |-binary-classification-46-features.csv

       |-mysql
   
          |-iot.sql

# import data to mysql
    mysql -hlocalhost -uroot -piot_admin iot < ./mysql/iot.sql
   
# Run command in scripts folder
   1. cd ./scripts
   
   2. sh create_dataset.sh  
   
   3. sh create_sspe_coap.sh
   
   4. sh create_sample.sh
   
   5. sh train_test.sh
   
# Check result with jupyter-notebooks.
