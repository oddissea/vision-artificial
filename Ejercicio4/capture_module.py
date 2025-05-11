# capture_module.py

import cv2
import glob
import os


class CapturaImagenes:
    def __init__(self, folder_path):
        """
        Inicializa la clase con el directorio de las imágenes.
        Se crean listas para almacenar los nombres e imágenes principales
        y las que se identifican como 'modelo'.
        """
        self.folder_path = folder_path
        self.nombres = []
        self.imagenes = []
        self.modelo_nombres = []
        self.modelo_imagenes = []

    def capturar(self):
        """
        Recorre todos los archivos del directorio especificado.
        Extrae el nombre del archivo (sin extensión) y reemplaza los guiones bajos por espacios.
        """
        # Se obtiene la lista de todos los archivos en el directorio
        for imagen_path in glob.glob(os.path.join(self.folder_path, '*')):
            # Extraer el nombre base del archivo (por ejemplo, "modelo_1.jpg" → "modelo_1")
            base_name = os.path.basename(imagen_path)
            name_without_ext, _ = os.path.splitext(base_name)

            # Reemplaza guiones bajos por espacios para obtener el título final
            imagen_nombre = name_without_ext.replace("_", " ")

            # Cargar la imagen en formato BGR
            imagen = cv2.imread(imagen_path)
            if imagen is None:
                # En caso de no poder cargar la imagen se salta
                continue
            # Convertir de BGR a RGB
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

            self.nombres.append(imagen_nombre)
            self.imagenes.append(imagen_rgb)

        # Imprime el conteo de imágenes principales y de modelos para verificación
        print(f"Imágenes: {len(self.nombres)}")

        # Devuelve un diccionario que separa las imágenes principales de las de modelo
        return {
            'originales': {
                'nombres': self.nombres,
                'imagenes': self.imagenes,
            }
        }


