# iTransformer time-series prediction api

Servicio backend para entrenamiento de Transformers

---

## instalación

```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo/backend

# Crea el entorno virtual
python -m venv venv

# Activa el entorno
.\venv\Scripts\activate

# Actualiza pip (para evitar errores de instalación)
.\venv\Scripts\python.exe -m pip install --upgrade pip

# Instala dependencias
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# Corre el proyecto
.\venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

## APIs Disponibles

- POST /train (Entrenamiento Base) Permite crear un modelo nuevo desde cero. Envías un bloque de datos históricos en JSON y el sistema genera automáticamente el archivo de pesos (.pth) y el escalador de datos (.pkl).

- POST /predict (Inferencia de Resultados) Es el endpoint principal para obtener pronósticos. Recibe una ventana de tiempo (por defecto los últimos 96 registros) y el nombre del modelo que deseas usar. Devuelve una lista de objetos con las fechas futuras y los valores predichos para cada variable.

- POST /fine_tuning (Ajuste Incremental) Diseñado para mantener el modelo actualizado sin reentrenarlo todo. Toma un modelo ya existente y le aplica una ráfaga corta de entrenamiento con datos muy recientes. Utiliza un Learning Rate extremadamente bajo para "refinar" el conocimiento del modelo sin corromper lo que ya aprendió originalmente.

- POST /upload_model (Importación) Facilita el despliegue de modelos externos. Permite subir un archivo comprimido .zip que contenga tanto el modelo como su scaler asociado.

- GET /download_model/{model_name} (Exportación) Permite extraer modelos del servidor. Al solicitar un modelo por su nombre, la API empaqueta los archivos .pth y .pkl en un solo .zip descargable.

---

## Arquitectura de referencia

Este proyecto es una implementación de API lista para producción basada en el modelo iTransformer, propuesto en el paper:

iTransformer: Inverted Transformers are Effective for Time Series Forecasting
🔗 Repositorio Original [THUML/iTransformer](https://github.com/thuml/iTransformer)
