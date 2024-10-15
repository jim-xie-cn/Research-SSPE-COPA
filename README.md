# Development environment prepare
1. setup python packages of ./install/requirements.txt
2. setup mysql，grant root access permission（user name is root，password is iot_admin)
# Data prepare
1. download dataset：
   https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications
   https://huggingface.co/datasets/JimXie/IIoTset/resolve/main/features.tar.gz
   https://huggingface.co/datasets/JimXie/IIoTset/resolve/main/mysql.tar.gz
   https://huggingface.co/datasets/JimXie/IIoTset/resolve/main/sample.tar.gz
   https://huggingface.co/datasets/JimXie/IIoTset/resolve/main/dataset.tar.gz
3. uncompress dataset to IIoTset folder
   after uncompressing, the struct of IIoTset folder should like below:
   IIoTset
       |-dataset
       |-features
       |-pcap-json
       |-dataset
       |-raw
       |-sample
       |-result
   
