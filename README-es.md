# AgroVision

API de clasificación de frescura de frutas construida con FastAPI y PyTorch. El modelo clasifica imágenes de frutas (manzana, banana, melón amargo, pimiento, naranja, tomate) como frescas o en mal estado.

## Estructura del Proyecto

```
agrovision/
├── artifacts/                  # Artefactos del modelo cargados al iniciar
│   ├── classes.json            # Definición de etiquetas de clase
│   ├── class_weights.json      # Pesos de clase usados durante el entrenamiento
│   └── normalization_stats.json # Media y desviación estándar para normalización
│
├── data/
│   └── raw/                    # Dataset de imágenes organizado por clase
│
├── notebooks/                  # Jupyter notebooks del pipeline de ML
│   ├── 00_data_acquisition.ipynb
│   ├── 01_data_ingestion.ipynb
│   ├── 02_eda_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/                        # Código fuente de la aplicación
│   ├── main.py                 # Punto de entrada de la app FastAPI
│   ├── config.py               # Configuración cargada desde variables de entorno / .env
│   └── features/
│       └── freshness/          # Feature de clasificación de frescura
│           ├── freshness_controller.py  # Definición de rutas
│           ├── freshness_service.py     # Lógica de inferencia del modelo
│           ├── dtos/                    # Objetos de transferencia de datos (respuesta)
│           └── pipes/                   # Procesamiento de solicitudes (decodificación de imagen)
│
├── tests/                      # Suite de pruebas con Pytest
│   ├── conftest.py
│   ├── test_main.py
│   └── features/
│       └── freshness/
│
├── requirements.txt            # Dependencias de producción
├── requirements-dev.txt        # Dependencias de desarrollo
└── .env                        # Variables de entorno (no incluido en el repositorio)
```

## Configuración

### 1. Crear y activar un entorno virtual

```bash
python -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Para desarrollo (incluye herramientas de prueba):

```bash
pip install -r requirements-dev.txt
```

### 3. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
MLFLOW_SERVER_URI=http://localhost:5000
MLFLOW_MODEL_URI=models:/<nombre-del-modelo>/<version-o-alias>
ALLOWED_ORIGINS=["http://localhost:3000"]
```

| Variable | Descripción |
|---|---|
| `MLFLOW_SERVER_URI` | URI del servidor MLflow en ejecución |
| `MLFLOW_MODEL_URI` | URI del modelo en MLflow para cargar el modelo registrado |
| `ALLOWED_ORIGINS` | Arreglo JSON con los orígenes CORS permitidos |

### 4. Obtener los artefactos del modelo

Asegúrate de que el directorio `artifacts/` contenga `classes.json` y `normalization_stats.json` antes de iniciar el servidor. Se pueden obtener mediante DVC:

```bash
dvc pull
```

## Ejecutar la API

```bash
fastapi run src/main.py
```

El servidor se inicia en `http://0.0.0.0:8000` por defecto.

Para desarrollo con recarga automática:

```bash
fastapi dev src/main.py
```

## Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Verificación de estado |
| `POST` | `/freshness/classify` | Clasifica una imagen de fruta como fresca o en mal estado |

La documentación interactiva de la API está disponible en `http://localhost:8000/docs` una vez que el servidor esté en ejecución.

### Ejemplo de solicitud

```bash
curl -X POST http://localhost:8000/freshness/classify \
  -F "file=@/ruta/a/imagen_fruta.jpg"
```

## Ejecutar Pruebas

```bash
pytest
```
