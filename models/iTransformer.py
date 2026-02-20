import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    Enlace al artículo original: https://arxiv.org/abs/2310.06625
    iTransformer: Los Transformers invertidos son efectivos para la predicción de series temporales.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # --- Configuración de dimensiones ---
        self.seq_len = configs.seq_len    # Longitud del historial (ventana de entrada)
        self.pred_len = configs.pred_len  # Longitud de la predicción (ventana de salida)
        self.output_attention = configs.output_attention # Si se desea extraer los pesos de atención
        self.use_norm = configs.use_norm  # Indica si se aplica normalización para series no estacionarias

        # --- Capa de Embedding (Invertida) ---
        # En el iTransformer, el embedding no se hace sobre el tiempo, sino sobre la serie completa de cada variable.
        # Transforma la serie de tiempo de cada variable en un vector de dimensión 'd_model'.
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        self.class_strategy = configs.class_strategy # Estrategia de clasificación (si aplica)

        # --- Arquitectura del Encoder ---
        # Se construye una pila de capas de Encoder (arquitectura "Encoder-only")
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers) # Repite según el número de capas definidas
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model) # Normalización de capa final del encoder
        )

        # --- Proyector Final ---
        # Una capa lineal que toma la representación del encoder y la proyecta a la longitud de predicción deseada.
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Lógica principal de predicción.
        x_enc: Datos históricos [Batch, Time, Variates]
        x_mark_enc: Marcas de tiempo (timestamps) de entrada
        """

        # 1. Normalización (Inspirada en el 'Non-stationary Transformer')
        # Ayuda a mitigar el cambio de distribución de los datos en el tiempo.
        if self.use_norm:
            # Calcula media y desviación estándar por cada serie en el batch
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Dimensiones: B = Batch size, L = Seq Len (Tiempo), N = Number of Variates (Variables)
        _, _, N = x_enc.shape

        # 2. Embedding Invertido
        # Proceso crucial: B L N -> B N E (donde E es d_model)
        # Cada variable (N) ahora se trata como un "Token" independiente que contiene toda su historia.
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # 3. Paso por el Encoder
        # Se aplica el mecanismo de Atención. Aquí las variables "conversan" entre sí
        # para entender cómo el cambio en una afecta a la otra.
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 4. Proyección y Reordenamiento
        # El proyector lleva los datos de la dimensión interna (E) a la dimensión de salida (S = pred_len)
        # B N E -> B N S -> Permutamos a -> B S N (Batch, Pred_Len, Variables)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        # 5. Des-Normalización
        # Devuelve los datos a su escala original (ej. de 0-1 a grados Celsius reales)
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Método de ejecución estándar de PyTorch.
        """
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # Si se activó la salida de atención, devuelve la predicción y los pesos de atención.
        if self.output_attention:
            # Retorna solo la ventana de predicción final [Batch, Pred_Len, Variates]
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]