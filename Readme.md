# EEG Classification
## Code Structure:
    Configurations: related configuration about EEG clissification model
    Data: It is a folder, which used to save raw dataset and processed dataset.
    Data process model：eeg_process.py, read EEG dataset，and split dataset into train, val and test set.
    Data upload model：eeg_dataset.py 
    Model：eeg_network.py
    Train：eeg_train.py

## Dataset
   The downlaod addresse of EEG datasets:  
   https://exhibits.stanford.edu/data/catalog/tc919dd5388

## Useage
   
   Run data process model to split data into train,val and test set.  
   ```
   python data_process --config_path ./configurations/eeg_config.yaml
   ```
   Train EEG classification model  
   ```
   python eeg_train.py --config_path ./configurations/eeg_config.yaml
   ```



## Reference
[1] Kaneshiro, Blair, et al. "A representational similarity analysis of the dynamics of object processing using single-trial EEG classification." Plos one 10.8 (2015): e0135697.  
[2] Yu, Bing, Haoteng Yin, and Zhanxing Zhu. "Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting." arXiv preprint arXiv:1709.04875 (2017).
