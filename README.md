# Detector de URLs spam con Flask

## Contexto

Este proyecto integra en una aplicación web Flask un modelo de Machine Learning entrenado previamente para clasificar URLs como `spam` o `no spam`.

El objetivo principal no es volver a entrenar el modelo, sino cargar el archivo guardado, recibir una URL desde un formulario web y mostrar la predicción al usuario.

## Modelo

El modelo utilizado es un pipeline con:

- `TF-IDF` para vectorizar el texto de la URL.
- `SVM` como clasificador.

Antes de predecir, la aplicación aplica el mismo preprocesamiento usado durante el entrenamiento:

```text
URL original -> preprocess_url() -> modelo -> spam / no spam
```

## Cómo ejecutar el proyecto en local

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

2. Entrar en la carpeta `src`:

```bash
cd src
```

3. Ejecutar la app:

```bash
python app.py
```

4. Abrir la aplicación en el navegador:

```text
http://127.0.0.1:5000
```

## Despliegue en Render

Configuración esperada:

- Root directory: `src`
- Start command: `gunicorn app:app`
- Python: `3.11`

## Archivos principales

- `src/app.py`: aplicación Flask.
- `src/templates/index.html`: plantilla HTML del formulario.
- `src/static/css/style.css`: estilos de la interfaz.
- `src/explore.ipynb`: notebook de integración y prueba del modelo.
- `models/svm_url_spam_classifier.joblib`: modelo entrenado.
- `requirements.txt`: dependencias del proyecto.
- `runtime.txt`: versión de Python para Render.

## Créditos

Proyecto realizado como parte del Bootcamp de Data Science y Machine Learning de 4Geeks Academy.
