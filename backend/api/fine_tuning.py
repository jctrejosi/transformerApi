import torch
import pickle
from torch.utils.data import DataLoader
from model.iTransformer import Model
from utils.convertJson import JSONDataset

def run_fine_tuning(data_json, config, model_path, scaler_path):
    # 1. Cargar Scaler existente (NO SE RE-ENTRENA)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # 2. Preparar datos con el scaler viejo
    dataset = JSONDataset(data_json, config, scaler=scaler)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 3. Cargar Modelo existente
    model = Model(config)
    model.load_state_dict(torch.load(model_path))
    
    # 4. LR MUY BAJO para no olvidar el pasado
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 
    criterion = torch.nn.MSELoss()

    model.train()
    for _ in range(2): # Solo 1 o 2 pasadas rápidas
        for batch_x, batch_y, batch_x_m, batch_y_m in loader:
            optimizer.zero_grad()
            outputs = model(batch_x, batch_x_m, torch.zeros_like(batch_y), batch_y_m)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), model_path)
    return "Fine-Tuning Complete"