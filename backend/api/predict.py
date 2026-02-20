import torch
import pandas as pd
import numpy as np
from model.iTransformer import Model
from utils.timefeatures import time_features

class WeatherPredictor:
    def __init__(self, model_path, config, scaler):
        """
        model_path: Ruta al archivo .pth
        config: Diccionario con la arquitectura
        scaler: El objeto StandardScaler cargado desde el entrenamiento
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = scaler
        
        # 1. Cargar Arquitectura
        self.model = Model(self.config).to(self.device)
        
        # 2. Cargar Pesos
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _process_input(self, json_data):
        df = pd.DataFrame(json_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # IMPORTANTE: Usamos el scaler que ya viene del entrenamiento masivo
        data_values = df.drop(columns=['date']).values
        data_scaled = self.scaler.transform(data_values)
        
        # Convertir a tensores [Batch, Seq_Len, Vars]
        x_enc = torch.FloatTensor(data_scaled).unsqueeze(0).to(self.device)
        
        # Generar marcas de tiempo
        # Usamos freq='t' para Bitcoin (minutos) o 'h' (horas)
        # Convertimos la columna a un DatetimeIndex explícito
        time_index = pd.DatetimeIndex(df['date'])
        x_mark_enc = torch.FloatTensor(time_features(time_index, freq='h')).transpose(0, 1).unsqueeze(0).to(self.device)
        
        return x_enc, x_mark_enc

    def predict(self, json_history, points_to_return=None):
        """
        Realiza una inferencia utilizando el modelo cargado.
        
        Args:
            json_history (list): Lista de diccionarios con los datos históricos (seq_len registros).
            points_to_return (int, optional): Cantidad de pasos futuros a devolver. 
                                              Si es None, devuelve el pred_len total.
        
        Returns:
            list: Lista de diccionarios donde cada elemento contiene las variables predichas 
                  y su marca de tiempo calculada.
        """
        # 1. Extraer nombres de columnas (excluyendo 'date') y la última fecha conocida
        df_temp = pd.DataFrame(json_history)
        column_names = [col for col in df_temp.columns if col != 'date']
        last_date = pd.to_datetime(df_temp['date'].iloc[-1])
        
        # 2. Preprocesar entrada (Escalamiento y Tensores)
        x_enc, x_mark_enc = self._process_input(json_history)
        p_len = self.config.pred_len
        
        with torch.no_grad():
            # 3. Preparar placeholders para el Decoder (ceros)
            # dec_inp: [Batch, Pred_Len, Num_Vars]
            dec_inp = torch.zeros([1, p_len, x_enc.shape[-1]]).to(self.device)
            # x_mark_dec: [Batch, Pred_Len, Num_TimeFeatures]
            x_mark_dec = torch.zeros([1, p_len, x_mark_enc.shape[-1]]).to(self.device)
            
            # 4. Inferencia del modelo
            outputs = self.model(x_enc, x_mark_enc, dec_inp, x_mark_dec)
            
            # 5. Desescalar predicciones a valores reales
            # outputs[0] extrae el primer (y único) batch
            predictions = self.scaler.inverse_transform(outputs.detach().cpu().numpy()[0])
            
            # 6. Recortar la ventana de predicción si se solicitó un número menor de puntos
            if points_to_return and points_to_return < p_len:
                predictions = predictions[:points_to_return]
            
            # 7. Formatear salida como lista de objetos con nombres de columnas y fechas
            final_forecast = []
            for i, row in enumerate(predictions):
                # Calcular la fecha del paso actual (asumiendo frecuencia horaria 'h')
                prediction_date = last_date + pd.Timedelta(hours=i + 1)
                
                # Crear el objeto de respuesta para este paso de tiempo
                item = {"date": prediction_date.strftime("%Y-%m-%d %H:%M:%S")}
                
                # Mapear cada valor numérico a su columna correspondiente
                for idx, col in enumerate(column_names):
                    item[col] = float(row[idx])
                
                final_forecast.append(item)
                
        return final_forecast