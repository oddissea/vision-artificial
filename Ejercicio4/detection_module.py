# detection_module.py

import cv2
import numpy as np


class Detector:
    def __init__(self,chain_approx=cv2.CHAIN_APPROX_SIMPLE):
        """
        Inicializa el detector con los parámetros de filtrado.

        :param chain_approx: Método de aproximación de contornos (por defecto, cv2.CHAIN_APPROX_SIMPLE).
        """
        self.chain_approx = chain_approx

    @staticmethod
    def compute_hull_ratio(contour):
        """
        Calcula el ratio: área del contorno / área del convex hull, y retorna también el convex hull.
        """
        hull = cv2.convexHull(contour)
        _M = cv2.moments(hull)
        if _M["m00"] != 0:
            cx = int(_M["m10"] / _M["m00"])
            cy = int(_M["m01"] / _M["m00"])
        else:
            cx, cy = 0, 0

        centroide_hull = (cx, cy)

        area_hull = cv2.contourArea(hull)
        area_contour = cv2.contourArea(contour)
        ratio = area_contour / area_hull if area_hull != 0 else 0
        return hull, area_hull, area_contour, ratio, centroide_hull

    def filter_duplicates(self, rect_list, threshold=None):
        """
        Agrupa detecciones cuyos centroides están a menos de 'threshold' píxeles,
        conservando la detección de menor área en cada grupo.
        Si no se especifica 'threshold', se usa self.centroid_threshold.
        """

        unique_rectangles = []
        for rect in rect_list:
            agregado = False
            for idx, u_rect in enumerate(unique_rectangles):
                dist = np.linalg.norm(np.array(rect['centroide']) - np.array(u_rect['centroide']))
                if dist < threshold:
                    agregado = True
                    if rect['area'] < u_rect['area']:
                        unique_rectangles[idx] = rect
                    break
            if not agregado:
                unique_rectangles.append(rect)
        return unique_rectangles

    def extract_detections(self, pre_img, mode=cv2.RETR_TREE,  centroid_threshold=20, min_area=500, max_area=10000):
        """
        Procesa una imagen para extraer detecciones:
          - Convierte a escala de grises (si es necesario)
          - Extrae contornos usando el modo 'mode'
          - Filtra por área utilizando el rectángulo delimitador
          - Agrupa duplicados por cercanía de centroides
          - Para cada detección, calcula el convex hull, el área del hull, el área del contorno y el ratio.

        Retorna un diccionario con:
            'detecciones': Lista de detecciones (cada una es un diccionario con la información mencionada)
            'gray': La imagen en escala de grises.
        """
        if len(pre_img.shape) > 2:
            gray = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = pre_img.copy()

        contours, _ = cv2.findContours(gray, mode, self.chain_approx)
        rect_list = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if min_area <= area <= max_area:
                centroide = (int(x + w / 2), int(y + h / 2))
                rect_list.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'centroide': centroide,
                    'contour': cnt,
                    'rect_ratio': w / h
                })

        unique_rectangles = self.filter_duplicates(rect_list, threshold=centroid_threshold)
        unique_rectangles = sorted(unique_rectangles, key=lambda r: r['area'], reverse=True)

        # Actualizar cada detección con información del hull y áreas
        for rect in unique_rectangles:
            cnt = rect['contour']
            (rect['hull'], rect['area_hull'],
             rect['area_contour'], rect['ratio'],
             rect['centroide_hull']) = self.compute_hull_ratio(cnt)
            rect['dist_centroides'] = np.sqrt(np.sum((np.array(rect['centroide_hull']) - np.array(rect['centroide']))**2))

        return {'detecciones': unique_rectangles, 'gray': gray, 'class': ""}

