# preprocess_module.py

import cv2
import numpy as np

class Preprocesador:
    def __init__(self, d_umbral=10, t_umbral=30, min_area=20, conn=8):
        self.d_umbral = d_umbral
        self.t_umbral = t_umbral
        self.min_area = min_area
        self.conn = conn

    @staticmethod
    def _gray(img):
        """Convierte la imagen a escala de grises."""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def _cleanup(self, img):
        """Realiza una limpieza previa de la imagen: elimina valores inferiores al umbral."""
        img_copy = img.copy()
        img_copy[img_copy < self.d_umbral] = 0
        return img_copy

    def _thresh(self, img):
        """Crea una máscara binaria global usando el umbral t_umbral."""
        _, mascara = cv2.threshold(img, self.t_umbral, 255, cv2.THRESH_BINARY)
        return mascara

    def _denoise(self, img):
        """Elimina pequeños blobs de la imagen binaria usando connectedComponentsWithStats."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, self.conn, cv2.CV_32S)
        mascara_filtrada = 255 * (labels > 0).astype(img.dtype)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                mascara_filtrada[labels == i] = 0
        return mascara_filtrada

    def procesar_imagen(self, imagen):
        """
        Pipeline de preprocesamiento original para imágenes con fondo relativamente uniforme.
        Devuelve un diccionario con:
          'gray', 'cleanup', 'thresh' y 'denoise'
        """
        img_gray = self._gray(imagen)
        img_cleanup = self._cleanup(255 - img_gray)
        img_thresh = self._thresh(img_cleanup)
        img_denoise = self._denoise(img_thresh)
        return {
            'gray': img_gray,
            'cleanup': img_cleanup,
            'thresh': img_thresh,
            'denoise': img_denoise
        }

    def procesar_imagen_gradiente(self, imagen):
        """
        Pipeline de preprocesamiento para imágenes con fondo degradado y que requieren detección de bordes.
        Los pasos son:
          1. Convertir a escala de grises.
          2. Estimar el fondo aplicando un desenfoque gaussiano (kernel grande).
          3. Corregir la iluminación dividiendo la imagen gris por el fondo estimado (flat‑field correction).
          4. Aplicar un detector de bordes (Canny) sobre la imagen corregida.
          5. Aplicar umbralización adaptativa a la imagen corregida.
          6. Realizar una operación morfológica (cierre) para unir rupturas.
          7. Aplicar la eliminación de pequeños blobs (denoise).
        Devuelve un diccionario con:
          :return 'gray': Imagen en escala de grises original.
          :return 'edges': Mapa de bordes obtenido con Canny.
          :return 'denoise': Imagen final después de denoise.
        """
        # 1. Convertir a escala de grises
        gray = self._gray(imagen)

        # 2. Estimar el fondo usando un desenfoque gaussiano (kernel grande, ajustar según imagen)
        kernel_size = (101, 101)
        background = cv2.GaussianBlur(gray, kernel_size, 0)

        # 4. Corrección de iluminación: dividir la imagen gray entre el fondo
        epsilon = 1e-6
        corrected = (gray.astype(np.float32) / (background.astype(np.float32) + epsilon)) * 128
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        # 5. Detector de bordes: aplicar Canny sobre la imagen corregida
        edges = cv2.Canny(corrected, 50, 120)

        # 5. Umbralización adaptativa sobre la imagen corregida
        thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # 6. Denoise: eliminar pequeños blobs
        denoise = self._denoise(thresh)

        return {
            'gray': gray,
            'background': background,
            'corrected': corrected,
            'edges': edges,
            'thresh': thresh,
            'denoise': denoise
        }


