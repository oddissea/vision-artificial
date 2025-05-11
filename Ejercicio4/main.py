import cv2
from capture_module import CapturaImagenes
from preprocess_module import Preprocesador
from detection_module import Detector
from representation_module import Visualizador
from classification_module import Clasificador


def capturar(path):
    captura = CapturaImagenes(path)
    images = captura.capturar()

    # Unificar la estructura en una lista de diccionarios.
    data = []
    for nombre, img in zip(images['originales']['nombres'], images['originales']['imagenes']):
        data.append({
            'nombre': nombre,
            'original': img
        })

    return data

def preprocesar(data):
    preprocesador = Preprocesador(d_umbral=10, t_umbral=30, min_area=20, conn=8)
    for item in data:
        if item['nombre'] == 'colores':
            preproc = preprocesador.procesar_imagen_gradiente(item['original'])
        else:
            preproc = preprocesador.procesar_imagen(item['original'])
        item['preprocesado'] = preproc

    print("Preprocesamiento completado para", len(data), "imágenes.")

def detectar(data):
    detector = Detector()
    for item in data:
        if item['nombre'] == 'colores':
            detecc = detector.extract_detections(item['preprocesado']['denoise'],
                                                 mode=cv2.RETR_TREE, min_area=100, max_area=1600,
                                                 centroid_threshold=10)
        else:
            detecc = detector.extract_detections(item['preprocesado']['denoise'],
                                                 mode=cv2.RETR_TREE, min_area=500, max_area=9000,
                                                 centroid_threshold=20)
        # 'detecc' es un diccionario con claves 'detecciones' y 'gray'
        item['detecciones'] = detecc['detecciones']
        item['detection_gray'] = detecc['gray']

    print("Detección completada para", len(data), "imágenes.")

def _caracteristicas(data, visualizador, path, solo_griega):
    # c) Visualización de detecciones sobre cada imagen (por ejemplo, dibujando rectángulos, hulls y contornos)
    for img_index, item in enumerate(data):
        img_original = item['original'].copy()
        detecciones = item['detecciones']  # Lista de detecciones con la información completa
        # Dibujar rectángulos
        img_rect = visualizador.dibujar_rect(img_original, detecciones, color=(0, 255, 0), thickness=2, solo_griega=solo_griega)
        # Dibujar convex hulls
        img_hull = visualizador.dibujar_hulls(img_original, detecciones, color=(255, 0, 0), thickness=2, solo_griega=solo_griega)
        # Dibujar contornos
        img_cont = visualizador.dibujar_contornos(img_original, detecciones, color=(0, 0, 255), thickness=2, solo_griega=solo_griega)
        # Mostrar cada uno en figuras separadas
        Visualizador([img_rect], [f"{item['nombre']} - Rectángulos"]).mostrar(images_por_fila=1, ancho_por_imagen=5, alto_por_imagen=5,
                                                                             save_path=f"{path}/{item['nombre']}_rect.png")
        Visualizador([img_hull], [f"{item['nombre']} - Hulls"]).mostrar(images_por_fila=1, ancho_por_imagen=5, alto_por_imagen=5,
                                                                        save_path=f"{path}/{item['nombre']}_hull.png")
        Visualizador([img_cont], [f"{item['nombre']} - Contornos"]).mostrar(images_por_fila=1, ancho_por_imagen=5, alto_por_imagen=5,
                                                                   save_path=f"{path}/{item['nombre']}_cont.png")

def visualizar_caracteristicas(data):
    # a) Visualizar las imágenes originales.
    visualizador_original = Visualizador(
        images_param=[item['original'] for item in data],
        titles_param=[item['nombre'] for item in data]
    )
    visualizador_original.mostrar(images_por_fila=3, ancho_por_imagen=5, alto_por_imagen=5,
                                  save_path="save/originales.png")

    # b) Visualizar las imágenes preprocesadas: usaremos la versión 'denoise'
    visualizador_denoise = Visualizador(
        images_param=[item['preprocesado']['denoise'] for item in data],
        titles_param=[item['nombre'] for item in data]
    )
    visualizador_denoise.mostrar(images_por_fila=3, ancho_por_imagen=5, alto_por_imagen=5,
                                 save_path="save/denoise.png")

    _caracteristicas(data, visualizador_original, "save", False)

def visualizar_claisficacion(data):

    # a) Visualizar las imágenes originales.
    visualizador_original = Visualizador(
        images_param=[item['original'] for item in data],
        titles_param=[item['nombre'] for item in data]
    )

    _caracteristicas(data, visualizador_original, "clasificadas", True)


def main(path):
    # 1. Captura: Obtener las imágenes originales y sus nombres.
    data_capture = capturar(path)


    # 2. Preprocesamiento: Procesar cada imagen original.
    preprocesar(data_capture)

    # 3. Detección: Aplicar el detector a la imagen 'denoise' preprocesada.
    detectar(data_capture)

    # 4. Visualización
    visualizar_caracteristicas(data_capture)

    # 5. Clasificación: Mostrar la lista de características de cada detección para cada imagen.
    # Se utiliza la clase Clasificador para imprimir las características en consola.
    clasificador = Clasificador(data_capture, 1.5)
    clasificador.mostrar_caracteristicas()

    # 6. Visualizar clasificación
    visualizar_claisficacion(data_capture)

if __name__ == "__main__":
    ruta = "pdf"
    main(ruta)
