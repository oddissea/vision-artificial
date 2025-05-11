# Técnicas Avanzadas de Visión Artificial

Este repositorio contiene la implementación de métodos para la detección y reconocimiento de objetos utilizando técnicas avanzadas de visión artificial, desarrollado como parte del Máster en Investigación en Inteligencia Artificial de la UNED.

## Descripción

El proyecto aborda varias áreas fundamentales del procesamiento y análisis de imágenes:

- **Segmentación basada en modelos**: Implementación y comparación de la transformada de Hough y RANSAC para detección de formas geométricas y ajuste de líneas en imágenes con ruido.
- **Descriptores de puntos característicos**: Exploración de detectores y descriptores como SIFT, ORB y Harris Corner para correspondencia entre imágenes y aplicaciones derivadas.
- **Técnicas de clasificación de objetos**: Desarrollo de sistemas para análisis de blobs, clasificadores Haar en cascada y combinación HoG+SVM.

## Estructura del proyecto

```
├── data/                  # Imágenes y datasets utilizados
├── modules/               # Módulos del sistema
│   ├── capture_module.py      # Lectura y preparación de imágenes
│   ├── preprocess_module.py   # Preprocesamiento y mejora de imágenes
│   ├── detection_module.py    # Detección de objetos y contornos
│   ├── classification_module.py # Clasificación de objetos detectados
│   └── representation_module.py # Visualización de resultados
├── notebooks/             # Notebooks para casos específicos
│   ├── hough_ransac.ipynb     # Comparativa de métodos de segmentación
│   ├── feature_descriptors.ipynb # Análisis de descriptores de características
│   ├── forms_alignment.ipynb  # Alineación de formularios
│   └── object_recognition.ipynb # Sistemas de reconocimiento
└── main.py                # Punto de entrada principal
```

## Aplicaciones implementadas

1. **Alineación y procesamiento de texto**: Detección de orientación del texto mediante transformada de Hough, transformación afín para alineación horizontal y segmentación de caracteres con normalización de tamaño.

2. **Procesamiento de formularios**: Sistema para transformar formularios rellenos a la misma escala y orientación que un patrón utilizando correspondencias ORB.

3. **Construcción de panoramas**: Implementación de stitching para unir múltiples imágenes en una vista panorámica mediante detección y emparejamiento de puntos característicos.

4. **Análisis de blobs**: Sistema para la distinción de diferentes tipos de objetos mediante análisis métrico de forma, tamaño, intensidad y textura.

5. **Detección facial**: Implementación de clasificadores en cascada con características Haar para detección de rostros y ojos en diferentes condiciones.

6. **Detección de peatones**: Sistema basado en la combinación de descriptores HoG con clasificadores SVM.

7. **Reconocimiento de formas específicas**: Desarrollo de un sistema para distinguir cruces griegas de cruces latinas en imágenes con diferentes niveles de ruido y deformación.

## Resultados y métricas

El proyecto incluye análisis comparativos de rendimiento entre los diferentes métodos implementados:

- **Transformada de Hough vs. RANSAC**: Evaluación de precisión y robustez frente al ruido.
- **Detección facial**: Análisis del impacto de parámetros como `scaleFactor` y condiciones de iluminación en la precisión.
- **Descriptores de características**: Evaluación de la efectividad de diferentes descriptores para aplicaciones de correspondencia y registro de imágenes.

## Dependencias

- Python 3.8+
- OpenCV 4.5+
- NumPy
- Matplotlib
- scikit-learn
- scikit-image

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/oddissea/vision-artificial.git
cd vision-artificial

# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

Para ejecutar la aplicación principal:

```bash
python main.py
```

Para casos específicos, puede explorar los notebooks en la carpeta `notebooks/`.

## Contribución

Las contribuciones son bienvenidas. Por favor, abra un issue para discutir grandes cambios antes de enviar un pull request.

## Licencia

[MIT](LICENSE)