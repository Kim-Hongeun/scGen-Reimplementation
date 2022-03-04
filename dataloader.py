# Data Loader

import os
import numpy as np
import pandas as pd
import torch
import anndata

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

class scGenDataset(Dataset):
    def __init__(self, anndata_path):

        self.anndata_path = anndata_path
        self.annData = anndata.read_h5ad(self.anndata_path)
        self.X = self.annData.to_df().to_numpy()
        self.cellType = self.annData.obs['cell_type']
        self.condition = self.annData.obs['condition']
        
        # Label Encode
        self.le = LabelEncoder()
        self.le.fit(self.cellType) 
        self.cellTypeEncoded = np.expand_dims(self.le.transform(self.cellType), axis=1)
        self.le.fit(self.condition)
        self.conditionEncoded = np.expand_dims(self.le.transform(self.condition), axis=1) 

        # Concat
        self.concat_x = np.concatenate([self.X, self.conditionEncoded, self.cellTypeEncoded], axis=1)

    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.concat_x[idx])
        return  x
    
def get_loader(anndata_path, batch_size, shuffle):

    scgenData = scGenDataset(anndata_path=anndata_path)

    data_loader = DataLoader(dataset=scgenData, 
                            batch_size=batch_size,
                            shuffle=shuffle)
    return data_loader 