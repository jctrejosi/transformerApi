import torch
import pickle
from torch.utils.data import DataLoader
from model.iTransformer import Model
from utils.convertJson import JSONDataset

def run_full_train(data_json, config, model_path, scaler_path):
    dataset = JSONDataset(data_json, config)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = Model(config).to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    for epoch in range(10): # Entrenamiento profundo
        for batch_x, batch_y, batch_x_m, batch_y_m in loader:
            optimizer.zero_grad()
            outputs = model(batch_x, batch_x_m, torch.zeros_like(batch_y), batch_y_m)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # GUARDAR TODO
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(dataset.scaler, f)
    return "Full Train Complete"