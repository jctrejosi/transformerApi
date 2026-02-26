# iTransformer time-series prediction api

Servicio backend para entrenamiento de Transformers

---

## Instalación local

```bash
git clone https://github.com/jctrejosi/transformerApi.git
cd transformerApi

# Crea el entorno virtual
python -m venv venv

# Activa el entorno
.\venv\Scripts\activate

# Actualiza pip (para evitar errores de instalación)
.\venv\Scripts\python.exe -m pip install --upgrade pip

# Instala dependencias
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# Corre el proyecto
.\venv\Scripts\python.exe -m uvicorn apis.main:app --reload
```

## Instalación con docker

```bash
# Construir la imagen
docker buildx build --no-cache -t transformer-api .

# Ejecutar contenedor
docker run -d -p 8000:8000 transformer-api
```

## Documentación de APIs

Esta API gestiona el flujo completo de modelos de series temporales, desde su creación inicial hasta la generación de pronósticos y exportación. El flujo de trabajo recomendado es: **Carga/Entrenamiento → Ajuste → Inferencia → Exportación.**

---

### 1. Gestión de modelos (Entrada)
Antes de generar predicciones, el sistema debe poseer un modelo base y su escalador de datos asociado.

#### `POST /train` (Entrenamiento Base)
Crea un modelo nuevo desde cero. Procesa datos históricos para generar el archivo de pesos (`.pth`) y el escalador (`.pkl`). Se ejecuta en segundo plano.

* **Entrada (JSON):**
    ```json
    {
      "model_name": "energia_v1",
      "data": [
        {"date": "2024-01-01 00:00:00", "OT": 12.5, "HU": 80},
        {"date": "2024-01-01 01:00:00", "OT": 12.8, "HU": 78}
      ]
    }
    ```
* **Respuesta:**
    ```json
    {
      "status": "training_started",
      "message": "El modelo 'energia_v1' se está entrenando en segundo plano.",
      "assets": ["saved_models/energia_v1.pth", "saved_models/energia_v1.pkl"]
    }
    ```

#### `POST /upload_model` (Importación)
Permite subir modelos entrenados externamente. Requiere un archivo `.zip` que contenga obligatoriamente un archivo `.pth` y un `.pkl`.

* **Entrada:** Archivo `multipart/form-data` (.zip).
* **Respuesta:**
    ```json
    {
      "status": "success",
      "message": "Modelo 'pack_externo.zip' instalado correctamente",
      "files": ["modelo_base.pth", "modelo_base.pkl"]
    }
    ```

---

### 2. Mantenimiento y actualización
Para evitar la degradación del modelo, se recomienda aplicar ajustes con datos nuevos.

#### `GET /list_models` (Inventario)
Devuelve un listado de todos los modelos disponibles en el servidor que están listos para ser usados (aquellos que tienen sus archivos `.pth` y `.pkl` completos).

* **Respuesta:**
    ```json
    {
      "models": [
        {
          "model_name": "energia_v1",
          "last_modified": "2024-03-20 15:30:00",
          "size_kb": 1250.5
        },
        {
          "model_name": "clima_test",
          "last_modified": "2024-03-21 09:15:22",
          "size_kb": 890.2
        }
      ],
      "count": 2
    }
    ```

#### `POST /fine_tuning` (Ajuste Incremental)
Aplica un entrenamiento rápido a un modelo existente usando datos muy recientes. Utiliza un *Learning Rate* bajo para refinar el conocimiento sin perder la base original.

* **Entrada (JSON):**
    ```json
    {
      "model_name": "energia_v1",
      "data": [
        {"date": "2024-03-01 10:00:00", "OT": 25.4, "HU": 40}
      ]
    }
    ```
* **Respuesta:**
    ```json
    {
      "status": "fine_tuning_started",
      "message": "Ajuste fino en marcha para el modelo 'energia_v1'."
    }
    ```

#### `DELETE /delete_model/{model_name}` (Eliminación)
Elimina permanentemente del servidor tanto los pesos del modelo (`.pth`) como su escalador (`.pkl`). Esta acción no se puede deshacer.

* **Parámetro de URL:** `model_name` (Nombre del modelo a borrar).
* **Ejemplo de llamada:** `DELETE /delete_model/modelo_obsoleto`
* **Respuesta:**
    ```json
    {
      "status": "success",
      "message": "Modelo 'modelo_obsoleto' eliminado correctamente.",
      "deleted_files": [
        "modelo_obsoleto.pth",
        "modelo_obsoleto.pkl"
      ]
    }
    ```

#### `DELETE /clear_all_models` (Limpieza Total Protegida)
Elimina **todos** los modelos y escaladores almacenados. Este endpoint está protegido por una clave de seguridad definida en las variables de entorno del servidor.

* **Encabezado Requerido:** `X-Admin-Key` (Tu clave de administrador).
* **Respuesta Exitosa (200):**
    ```json
    {
      "status": "success",
      "message": "Limpieza completa realizada con éxito."
    }
    ```
* **Respuesta Error (401):**
    ```json
    {
      "detail": "No autorizado. Clave de administración incorrecta o ausente."
    }
    ```

---

### 3. Inferencia de resultados
Una vez el modelo está cargado y entrenado, se utiliza para proyectar valores futuros.

#### `POST /predict` (Generación de Pronóstico)
Recibe una ventana de datos históricos (por defecto 96 registros) y devuelve la predicción para los puntos futuros especificados.

* **Entrada (JSON):**
    ```json
    {
      "model_name": "energia_v1",
      "points": 24,
      "data": [...] // Lista de registros históricos (seq_len)
    }
    ```
* **Respuesta:**
    ```json
    {
      "status": "success",
      "model_used": "energia_v1",
      "forecast": [
        {"date": "2024-03-02 00:00:00", "OT": 15.2, "HU": 70},
        {"date": "2024-03-02 01:00:00", "OT": 14.9, "HU": 72}
      ]
    }
    ```

---

### 4. Exportación (Salida)
Facilita la portabilidad de los modelos generados en el servidor.

#### `POST /download_model` (Exportación)
Solicita un modelo por su nombre para obtener un paquete `.zip` con los pesos y el escalador listos para usar en otro entorno.

* **Entrada (JSON):**
    ```json
    { "model_name": "energia_v1" }
    ```
* **Respuesta:** Descarga directa de archivo `energia_v1_complete.zip`.

---

## Arquitectura de referencia

Este proyecto es una implementación de API lista para producción basada en el modelo iTransformer, propuesto en el paper:

iTransformer: Inverted Transformers are Effective for Time Series Forecasting
🔗 Repositorio Original [THUML/iTransformer](https://github.com/thuml/iTransformer)
