# classification_viewer.py

class Clasificador:
    def __init__(self, data, umbral):
        """
        Inicializa el clasificador con la estructura unificada de datos.
        :param data: Una lista de diccionarios, cada uno con las claves:
            'nombre', 'original', 'preprocesado' y 'detecciones'.
        """
        self.data = data
        self.umbral = umbral

    def mostrar_caracteristicas(self):
        """
        Itera sobre cada imagen y muestra, en consola, las características
        de cada detección:
          - Área
          - Centroide
          - Contorno (se muestra su forma o número de puntos)
          - rect_ratio (relación ancho/alto del rectángulo delimitador)
          - Hull (se muestra la forma del array)
          - Centroide del hull
          - Área del hull
          - Área del contorno
          - Ratio (área_contour/area_hull)
          - Distancia entre centroides (por ejemplo, 'dist_centroides')
        """
        for item in self.data:
            print(f"Imagen: {item['nombre']} - Items detectados: {len(item['detecciones'])}")
            # Verifica si hay detecciones
            for idx, det in enumerate(item['detecciones'], start=1):
                # Calcular la clasificación basada en 'dist_centroides'
                dist = det.get('dist_centroides')
                if dist is not None:
                    # Se asigna "griega" si dist <= 1; de lo contrario "no_griega"
                    clasificacion = "griega" if dist <= self.umbral else "no_griega"
                else:
                    clasificacion = "N/A"
                det['class'] = clasificacion  # Añadimos la clave 'class' al diccionario de la detección

                print(f"  Elemento {idx}:")
                print(f"    Área: {det.get('area', 'N/A')}")
                print(f"    Centroide: {det.get('centroide', 'N/A')}")
                # Para el contorno mostramos su forma (por ejemplo, cantidad de puntos)
                contour = det.get('contour')
                if contour is not None:
                    print(f"    Contour: shape {contour.shape}")
                else:
                    print("    Contour: N/A")
                print(f"    Rect_ratio: {det.get('rect_ratio', 'N/A')}")
                hull = det.get('hull')
                if hull is not None:
                    print(f"    Hull: shape {hull.shape}")
                else:
                    print("    Hull: N/A")
                print(f"    Centroide hull: {det.get('centroide_hull', 'N/A')}")
                print(f"    Área hull: {det.get('area_hull', 'N/A')}")
                print(f"    Área contorno: {det.get('area_contour', 'N/A')}")
                print(f"    Ratio: {det.get('ratio', 'N/A')}")
                print(f"    Distancia centroides: {det.get('dist_centroides', 'N/A')}")
                print(f"    Clase: {det.get('class', 'N/A')}")
            print("-" * 40)


if __name__ == "__main__":
    # Ejemplo simulado de datos unificados
    import numpy as np
    ejemplo_data = [
        {
            'nombre': "Imagen 1",
            'original': np.zeros((100,100,3), dtype=np.uint8),
            'preprocesado': {
                'gray': np.zeros((100,100), dtype=np.uint8),
                'cleanup': np.zeros((100,100), dtype=np.uint8),
                'thresh': np.zeros((100,100), dtype=np.uint8),
                'denoise': np.zeros((100,100), dtype=np.uint8)
            },
            'detecciones': {
                'detecciones': [
                    {
                        'bbox': (10, 10, 50, 50),
                        'area': 2500,
                        'centroide': (35,35),
                        'contour': np.random.randint(0,255,(10,1,2)),
                        'rect_ratio': 1.0,
                        'hull': np.random.randint(0,255,(8,1,2)),
                        'centroide_hull': (40,40),
                        'area_hull': 3000,
                        'area_contour': 2500,
                        'ratio': 0.83,
                        'dist_centroides': 5.0
                    },
                    {
                        'bbox': (60, 60, 30, 30),
                        'area': 900,
                        'centroide': (75,75),
                        'contour': np.random.randint(0,255,(12,1,2)),
                        'rect_ratio': 1.0,
                        'hull': np.random.randint(0,255,(10,1,2)),
                        'centroide_hull': (78,78),
                        'area_hull': 1000,
                        'area_contour': 900,
                        'ratio': 0.9,
                        'dist_centroides': 3.0
                    }
                ],
                'gray': np.zeros((100,100), dtype=np.uint8),
                'class': "griega/no_griega"
            }
        }
    ]
    clasificador = Clasificador(ejemplo_data, 1.0)
    clasificador.mostrar_caracteristicas()
