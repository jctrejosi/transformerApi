import torch


class TriangularCausalMask():
    """
    Máscara Causal Triangular: 
    Se asegura de que en cada paso de tiempo, el modelo solo pueda atender 
    a los puntos pasados y al actual, bloqueando completamente el acceso al futuro.
    Es la máscara estándar de los Transformers autorregresivos.
    """
    def __init__(self, B, L, device="cpu"):
        # B: Batch size (Tamaño del lote)
        # L: Length (Longitud de la secuencia)
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # torch.triu crea una matriz triangular superior.
            # Al usar diagonal=1, deja la diagonal principal en 0 y pone 1s arriba.
            # Los 1s (True) indicarán dónde se debe aplicar el filtro de "infinito negativo".
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    """
    Máscara para ProbSparse Attention:
    Esta es más compleja. Se utiliza específicamente con el algoritmo 'ProbAttention' 
    (del modelo Informer). Su trabajo es filtrar la información del futuro pero 
    solo para los subconjuntos de puntos (índices) que el algoritmo seleccionó como importantes.
    """
    def __init__(self, B, H, L, index, scores, device="cpu"):
        # B: Batch size, H: Heads (Cabezales), L: Longitud original
        # index: Los índices de las consultas (queries) seleccionadas como más importantes.
        # scores: Los puntajes de atención calculados.
        
        # 1. Creamos una máscara triangular superior básica.
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        
        # 2. Expandimos la máscara para que coincida con el lote y los cabezales de atención.
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        
        # 3. Seleccionamos de la máscara completa solo las posiciones que coinciden con 
        # los índices de las 'queries' activas que pasaron el filtro de probabilidad.
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        
        # 4. Reestructuramos la máscara para que tenga la misma forma que la matriz de puntajes (scores).
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask