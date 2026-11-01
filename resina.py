import cv2
import numpy as np
import random
from pathlib import Path

#1.  Tomo imagen limpia. Entiendo su alto y su ancho
#ruta = "sin_defectos/M9317/20260303133705_C1_22.bmp"
#ruta = "sin_defectos/B3002/20251216125652_42.bmp"
ruta = "sin_defectos/f2273/20260222162425_C2_165.bmp"
#ruta ="sin_defectos/f6012/20260524092704_C4_14.bmp"
#ruta ="sin_defectos/M6321/20260113193019_04.bmp"

nombre_imagen = Path(ruta).stem          # nombre del archivo sin carpeta ni extension. Ej: 20260303133705_C1_22
carpeta_modelo = Path(ruta).parent.name  # nombre de la carpeta donde esta la imagen. Ej: M9317

#1.1  Seteo constantes
GROSOR_MIN = 0.003    # % grosor minimo de la linea
GROSOR_MAX = 0.015    # % grosor maximo de la linea
LARGO_MIN = 0.7       # % largo minimo de la linea
TRAMO = 20            # cada cuantos pixeles la linea puede subir o bajar un poco
MAX_INTENTOS = 5      # cantidad de imagenes a generar

#1.2  Seleccionar imagen y obtener sus propiedades
imagen = cv2.imread(ruta)
alto = imagen.shape[0]
ancho = imagen.shape[1]
lado_corto = min(alto,ancho)  #tomaremos el lado mas pequeño de la imagen para calcular que el grosor de la linea sea proporcional al tamaño de la imagen

#Crear la carpeta de resultados si no existe (cv2.imwrite no crea carpetas y no avisa si falla)
carpeta_salida = Path("resultados") / carpeta_modelo
carpeta_salida.mkdir(parents=True, exist_ok=True)

intentos = 0

while intentos < MAX_INTENTOS:
    intentos = intentos + 1

    # 2. Elegir al azar como sera la linea (siempre horizontal)
    grosor = max(1, int(lado_corto * random.uniform(GROSOR_MIN, GROSOR_MAX)))   # grosor de la linea en pixeles
    y = random.randint(0, alto - 1)   #los pixeles de una imagen se cuentan empezando desde 0. Una imagen de 400 px, empieza en la fila 0 y termina en la 399                                    # a que altura pasa la linea
    largo = int(ancho * random.uniform(LARGO_MIN, 1))                     # cuanto mide la linea
    inicio = random.randint(0, ancho - largo)                             # donde empieza (x)
    fin = inicio + largo                                                  # donde termina (x)
    intensidad = random.uniform(0.2, 0.5)   # cuanto aclara: 0.3 = acerca el pixel un 30% al blanco
    print("imagen", intentos, "- altura:", y, "- grosor:", grosor, "- intensidad:", round(intensidad, 2))

    # 3. Dibujar la linea blanca en una imagen negra (el "molde")
    molde = np.zeros(imagen.shape[:2], dtype=np.float32)
    cv2.line(molde, (inicio, y), (fin, y), 1.0, grosor)   # 1.0 color blanco

    # 4. Difuminar el borde
    molde = cv2.GaussianBlur(molde, (0, 0), max(1, grosor / 3))

    # 5. Aclarar la foto donde esta el molde
    #    (255 - imagen) es lo que le falta a cada pixel para llegar a blanco. Le sumamos una parte de eso
    resultado = imagen + (255 - imagen) * (intensidad * molde[..., None])

    # 6. Si la mascara sale vacia, apuntarlo en un archivo para revisarlo
    if molde.max() == 0:
        with open("resultados/mascaras_vacias.txt", "a") as log:
            log.write(f"{nombre_imagen}, resina {intentos}, grosor={grosor}\n")
        print("Aviso: máscara vacía, apuntada en mascaras_vacias.txt")

    # 7. Guardar imagenes
    cv2.imwrite(f"{carpeta_salida}/{nombre_imagen}_resina_{intentos}.png", resultado.astype(np.uint8))
    cv2.imwrite(f"{carpeta_salida}/{nombre_imagen}_resina_mascara_{intentos}.png", (molde * 255).astype(np.uint8))