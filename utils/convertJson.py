import torch
import pandas as pd
from torch.utils.data import Dataset
from utils.timefeatures import time_features
from utils.tools import StandardScaler

class JSONDataset(Dataset):
    def __init__(self, data_json, config, scaler=None):
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        
        # 1. Convertir JSON a DataFrame y ordenar por fecha
        df = pd.DataFrame(data_json)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # 2. Separar valores numéricos
        df_data = df.drop(columns=['date'])
        
        # 3. Manejo del Scaler (Normalización)
        if scaler is None:
            # Si estamos en /train, creamos un scaler nuevo
            self.scaler = StandardScaler(mean=df_data.values.mean(0), std=df_data.values.std(0))
        else:
            # Si estamos en /fine_tuning, usamos el que ya existía
            self.scaler = scaler
            
        data = self.scaler.transform(df_data.values)
        
        # 4. Características de tiempo (Marcas para el Transformer)
        # Usamos freq='h', cámbialo a 't' si tus datos son por minuto
        # Convertimos la columna 'date' en el índice de tiempo que espera la función
        data_stamp = time_features(df['date'].dt, freq='h').transpose(1, 0)

        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # Punto donde termina la historia y empieza la predicción
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y), \
               torch.FloatTensor(seq_x_mark), torch.FloatTensor(seq_y_mark)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1