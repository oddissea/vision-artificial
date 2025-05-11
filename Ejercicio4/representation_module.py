# representation_module.py

import matplotlib.pyplot as plt
import cv2
import numpy as np

class Visualizador:
    def __init__(self, images_param, titles_param=None):
        """
        Inicializa el visualizador con una lista de imágenes y, opcionalmente, títulos.
        :param images_param: Lista de imágenes (por ejemplo, arrays NumPy en formato RGB).
        :param titles_param: Lista opcional de títulos para cada imagen.
        """
        self.images = images_param
        if titles_param is None:
            self.titles = [""] * len(images_param)
        else:
            self.titles = titles_param

    def mostrar(self, images_por_fila=1, ancho_por_imagen=5, alto_por_imagen=5, save_path=None):
        """
        Muestra las imágenes en una cuadrícula.
        El tamaño de la figura se calcula en función del número de columnas y filas.
        :param images_por_fila: Número de imágenes por fila.
        :param ancho_por_imagen: Ancho (en pulgadas) asignado a cada imagen.
        :param alto_por_imagen: Alto (en pulgadas) asignado a cada imagen.
        :param save_path: Si se especifica, guarda la figura en la ruta dada.
        """
        total = len(self.images)
        filas = (total + images_por_fila - 1) // images_por_fila
        fig_size = (images_por_fila * ancho_por_imagen, filas * alto_por_imagen)
        fig, axes = plt.subplots(filas, images_por_fila, figsize=fig_size)

        # Asegurarse de que axes sea un arreglo 1D para indexar uniformemente
        if filas == 1 and images_por_fila == 1:
            axes = [axes]
        elif filas == 1 or images_por_fila == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i in range(total):
            ax = axes[i]
            ax.imshow(self.images[i], cmap='gray')
            ax.set_title(self.titles[i])
            ax.axis("off")
        for j in range(total, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    @staticmethod
    def dibujar_rect(image, detecciones, color=(0, 255, 0), thickness=1, solo_griega=False):
        """
        Dibuja sobre la imagen los rectángulos delimitadores (bbox) de cada detección.
        Si solo_griega es True, dibuja solo aquellas detecciones cuya clave 'class' sea "griega".
        :param image: Imagen original (RGB).
        :param detecciones: Lista de detecciones, cada una debe tener la clave 'bbox' y opcionalmente 'class'.
        :param color: Color del rectángulo (BGR).
        :param thickness: Grosor de la línea.
        :param solo_griega: Bandera para dibujar solo detecciones clasificadas como "griega".
        :return: Imagen con rectángulos (y centroides) dibujados.
        """
        imagen_copy = image.copy()
        for det in detecciones:
            if solo_griega and det.get('class', None) != "griega":
                continue
            x, y, w, h = det.get('bbox', (0, 0, 0, 0))
            cv2.rectangle(imagen_copy, (x, y), (x + w, y + h), color, thickness)
            cv2.circle(imagen_copy, det['centroide'], 1, (255, 0, 0), thickness)
            if 'centroide_hull' in det:
                cv2.circle(imagen_copy, det['centroide_hull'], 1, (0, 0, 255), thickness)
        return imagen_copy

    @staticmethod
    def dibujar_hulls(image, detecciones, color=(255, 0, 0), thickness=1, solo_griega=False):
        """
        Dibuja sobre la imagen el convex hull de cada detección.
        Si solo_griega es True, dibuja solo aquellas detecciones clasificadas como "griega".
        :param image: Imagen original (RGB).
        :param detecciones: Lista de detecciones, cada una debe tener la clave 'hull' y opcionalmente 'class'.
        :param color: Color del hull (BGR).
        :param thickness: Grosor de la línea.
        :param solo_griega: Bandera para dibujar solo detecciones clasificadas como "griega".
        :return: Imagen con convex hulls dibujados.
        """
        imagen_copy = image.copy()
        for det in detecciones:
            if solo_griega and det.get('class', None) != "griega":
                continue
            if 'hull' in det:
                cv2.drawContours(imagen_copy, [det['hull']], -1, color, thickness)
                cv2.circle(imagen_copy, det['centroide'], 1, (255, 0, 0), thickness)
                if 'centroide_hull' in det:
                    cv2.circle(imagen_copy, det['centroide_hull'], 1, (0, 0, 255), thickness)
        return imagen_copy

    @staticmethod
    def dibujar_contornos(image, detecciones, color=(0, 0, 255), thickness=1, solo_griega=False):
        """
        Dibuja sobre la imagen el contorno original de cada detección.
        Si solo_griega es True, dibuja solo aquellas detecciones clasificadas como "griega".
        :param image: Imagen original (RGB).
        :param detecciones: Lista de detecciones, cada una debe tener la clave 'contour' y opcionalmente 'class'.
        :param color: Color del contorno (BGR).
        :param thickness: Grosor de la línea.
        :param solo_griega: Bandera para dibujar solo detecciones clasificadas como "griega".
        :return: Imagen con contornos dibujados.
        """
        imagen_copy = image.copy()
        for det in detecciones:
            if solo_griega and det.get('class', None) != "griega":
                continue
            cv2.drawContours(imagen_copy, [det['contour']], -1, color, thickness)
            cv2.circle(imagen_copy, det['centroide'], 1, (255, 0, 0), thickness)
            if 'centroide_hull' in det:
                cv2.circle(imagen_copy, det['centroide_hull'], 1, (0, 0, 255), thickness)
        return imagen_copy

    @staticmethod
    def dibujar_centroides(image, detecciones, color=(0, 0, 255), radius=3, thickness=-1, solo_griega=False):
        """
        Dibuja sobre la imagen los centroides de cada detección.
        Si solo_griega es True, dibuja solo aquellas detecciones clasificadas como "griega".
        :param image: Imagen original (RGB).
        :param detecciones: Lista de detecciones, cada una debe tener la clave 'centroide' y opcionalmente 'class'.
        :param color: Color del círculo (BGR).
        :param radius: Radio del círculo.
        :param thickness: Grosor de la línea (-1 para relleno).
        :param solo_griega: Bandera para dibujar solo detecciones clasificadas como "griega".
        :return: Imagen con centroides dibujados.
        """
        imagen_copy = image.copy()
        for det in detecciones:
            if solo_griega and det.get('class', None) != "griega":
                continue
            if 'centroide' in det:
                cv2.circle(imagen_copy, det['centroide'], radius, color, thickness)
            if 'centroide_hull' in det:
                cv2.circle(imagen_copy, det['centroide_hull'], radius, (255, 255, 0), thickness)
        return imagen_copy

