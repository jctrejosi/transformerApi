import torch
import pandas as pd
import pickle
import os
from fastapi import APIRouter, HTTPException
from models.iTransformer import Model
from utils.timefeatures import time_features

router = APIRouter(tags=["prediction"])

class WeatherPredictor:
    def __init__(self, model_path, config, scaler):
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
        
        data_values = df.drop(columns=['date']).values
        data_scaled = self.scaler.transform(data_values)
        
        x_enc = torch.FloatTensor(data_scaled).unsqueeze(0).to(self.device)
        
        time_index = pd.DatetimeIndex(df['date'])
        x_mark_enc = torch.FloatTensor(time_features(time_index, freq='h')).transpose(0, 1).unsqueeze(0).to(self.device)
        
        return x_enc, x_mark_enc

    def predict(self, json_history, points_to_return=None):
        df_temp = pd.DataFrame(json_history)
        column_names = [col for col in df_temp.columns if col != 'date']
        last_date = pd.to_datetime(df_temp['date'].iloc[-1])
        
        x_enc, x_mark_enc = self._process_input(json_history)
        p_len = self.config.pred_len
        
        with torch.no_grad():
            dec_inp = torch.zeros([1, p_len, x_enc.shape[-1]]).to(self.device)
            x_mark_dec = torch.zeros([1, p_len, x_mark_enc.shape[-1]]).to(self.device)
            
            outputs = self.model(x_enc, x_mark_enc, dec_inp, x_mark_dec)
            
            predictions = self.scaler.inverse_transform(outputs.detach().cpu().numpy()[0])
            
            if points_to_return and points_to_return < p_len:
                predictions = predictions[:points_to_return]
            
            final_forecast = []
            for i, row in enumerate(predictions):
                prediction_date = last_date + pd.Timedelta(hours=i + 1)
                item = {"date": prediction_date.strftime("%Y-%m-%d %H:%M:%S")}
                for idx, col in enumerate(column_names):
                    item[col] = float(row[idx])
                final_forecast.append(item)
                
        return final_forecast

# --- ENDPOINT DE PREDICCIÓN ---
@router.post("/predict")
async def get_prediction(payload: dict):
    from api.main import config  # Importación para evitar círculos
    
    try:
        model_id = payload.get('model_name', 'default_model')
        target_model = f"saved_models/{model_id}.pth"
        target_scaler = f"saved_models/{model_id}.pkl"

        if not os.path.exists(target_model) or not os.path.exists(target_scaler):
            raise HTTPException(
                status_code=404, 
                detail=f"El modelo '{model_id}' no existe o no tiene scaler."
            )

        # Cargar el scaler específico del modelo solicitado
        with open(target_scaler, 'rb') as f:
            scaler = pickle.load(f)
        
        # Inicializar el predictor con el modelo solicitado
        current_predictor = WeatherPredictor(target_model, config, scaler)
        
        historical_data = payload.get('data')
        n_points = payload.get('points', config.pred_len)

        # Validación de longitud de datos
        if not historical_data or len(historical_data) < config.seq_len:
            raise HTTPException(
                status_code=400,
                detail=f"Se requieren exactamente {config.seq_len} registros históricos."
            )

        # Si mandan más de los necesarios, tomamos los últimos
        if len(historical_data) > config.seq_len:
            historical_data = historical_data[-config.seq_len:]

        forecast = current_predictor.predict(historical_data, points_to_return=n_points)
        
        return {
            "status": "success",
            "model_used": model_id,
            "forecast": forecast
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))