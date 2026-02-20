import numpy as np
import torch
import matplotlib.pyplot as plt

# 'agg' permite generar imágenes sin necesidad de una interfaz gráfica (ideal para servidores/APIs)
plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch, args):
    """
    Ajusta la tasa de aprendizaje (Learning Rate) durante el entrenamiento.
    Reduce el LR para que el modelo pueda converger mejor en las etapas finales.
    """
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Actualizando learning rate a {}'.format(lr))

class EarlyStopping:
    """
    Parada Temprana: Detiene el entrenamiento si la pérdida de validación deja
    de mejorar después de 'patience' épocas. Evita el sobreajuste (overfitting).
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Contador EarlyStopping: {self.counter} de {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """Guarda el modelo cuando la pérdida de validación disminuye."""
        if self.verbose:
            print(f'Pérdida de validación disminuyó ({self.val_loss_min:.6f} --> {val_loss:.6f}). Guardando modelo...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """
    Permite acceder a los diccionarios con puntos: dict.propiedad en lugar de dict['propiedad'].
    Muy útil para manejar los argumentos de configuración (args).
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    """
    Normalizador Estándar: Escala los datos para que tengan media 0 y desviación estándar 1.
    ¡CRÍTICO!: El iTransformer necesita que los datos estén normalizados para funcionar.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """Devuelve los datos a sus unidades originales (ej: de 0.5 a 25°C)."""
        return (data * self.std) + self.mean

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Genera un gráfico comparando la realidad (GroundTruth) con la predicción.
    """
    plt.figure()
    plt.plot(true, label='Realidad', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Predicción', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def adjustment(gt, pred):
    """
    Ajuste de anomalías: Si se detecta una anomalía en un segmento,
    trata todo el segmento como anómalo. Común en detección de intrusiones o fallos.
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0: break
                else:
                    if pred[j] == 0: pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0: break
                else:
                    if pred[j] == 0: pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)