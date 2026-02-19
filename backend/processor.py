import torch
import torch.nn as nn

class ITransformer(nn.Module):
    def __init__(self, num_variates, lookback, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        
        # 1. Proyección Lineal: Convierte la serie de tiempo de cada variable en un vector
        # No importa si el lookback es 48 o 100, se proyecta a d_model (128)
        self.enc_embedding = nn.Linear(lookback, d_model)
        
        # 2. El Encoder del Transformer: Analiza la relación entre variables
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 3. Proyección de Salida: Convierte el vector de vuelta a una predicción
        self.projection = nn.Linear(d_model, 1) # Predecimos 1 siguiente valor (t+1)

    def forward(self, x):
        # x shape: [Batch, Lookback, Num_Variates] -> ej: [1, 48, 3]
        
        # Permutamos para que las variables sean los "tokens"
        # Nuevo shape: [Batch, Num_Variates, Lookback]
        x = x.permute(0, 2, 1)
        
        # Proyectamos cada variable a la dimensión del transformer
        x = self.enc_embedding(x) # [Batch, Num_Variates, d_model]
        
        # Pasamos por el Transformer (Atención entre variables)
        x = self.transformer_encoder(x)
        
        # Proyectamos al futuro (t+1)
        x = self.projection(x) # [Batch, Num_Variates, 1]
        
        return x.squeeze(-1) # Retorna [Batch, Num_Variates]