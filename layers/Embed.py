import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    """
    Codificación Posicional: Como el Transformer procesa todo en paralelo, no sabe qué dato va
    antes o después. Esta clase inyecta información sobre el orden de la secuencia usando senos y cosenos.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Crear matriz de ceros para las posiciones
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False # No necesita entrenamiento

        # Crear vector de posiciones [0, 1, 2, ..., max_len]
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # Factor de división para las frecuencias de las ondas
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        # Aplicar seno a las posiciones pares e índice impar a las posiciones coseno
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # Se guarda como estado del modelo pero no como parámetro entrenable

    def forward(self, x):
        # Devuelve la codificación posicional ajustada al tamaño de la secuencia de entrada
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """
    Incrustación de Tokens: Convierte los valores numéricos brutos en una representación
    más rica (vectorial) usando una convolución de 1D.
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        # El padding cambia según la versión de torch para mantener la consistencia
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # Convolución para extraer características locales de la serie temporal
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # Inicialización de pesos para mejorar la convergencia
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # Cambia dimensiones para la convolución: [B, L, C] -> [B, C, L]
        # Y regresa a: [B, L, D]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """
    Incrustación Fija: Similar a PositionalEmbedding pero se usa para categorías
    temporales fijas como el día o el mes.
    """
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    Incrustación Temporal: Toma datos de tiempo (mes, día, hora, etc.) y los convierte en vectores.
    Asume que la entrada tiene 5 columnas de tiempo específicas.
    """
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        # Tamaños máximos para cada categoría temporal
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        # Se extrae cada componente del tiempo del tensor x_mark
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        # Se suman todas las representaciones temporales
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """
    Incrustación de Características de Tiempo: Una alternativa más simple que usa
    una capa lineal para procesar timestamps normalizados.
    """
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        # Mapa de dimensiones según la frecuencia (hora, minuto, segundo, etc.)
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    """
    Incrustación de Datos Estándar (Vanilla Transformer):
    Suma el valor del dato + su posición + su información temporal.
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            # Si no hay datos de fecha, solo sumamos valor y posición
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            # Suma de las tres fuentes de información
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    """
    Incrustación Invertida (ESPECIAL PARA iTRANSFORMER):
    Este es el cambio clave del paper. No usa codificación posicional clásica.
    Toma la serie completa de una variable y la proyecta linealmente.
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        # c_in aquí es en realidad seq_len (la longitud del tiempo)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # El cambio mágico: [Batch, Time, Variate] -> [Batch, Variate, Time]
        x = x.permute(0, 2, 1)

        if x_mark is None:
            # Proyecta la dimensión del tiempo a d_model
            x = self.value_embedding(x)
        else:
            # Permite concatenar variables externas (como el tiempo) como si fueran tokens adicionales
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))

        # El resultado es un conjunto de tokens donde cada token es una variable completa
        # x: [Batch, Variate, d_model]
        return self.dropout(x)