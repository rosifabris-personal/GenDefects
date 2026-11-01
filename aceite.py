import cv2
import numpy as np
import random
from pathlib import Path

#1.  Tomo imagen limpia. Entiendo su alto y su ancho
ruta = "sin_defectos/M9317/20260303133705_C1_22.bmp"
#ruta = "sin_defectos/B3002/20251216125652_42.bmp"
#ruta = "sin_defectos/f2273/20260222162425_C2_165.bmp"
#ruta ="sin_defectos/f6012/20260524092704_C4_14.bmp"
#ruta ="sin_defectos/M6321/20260113193019_04.bmp"


nombre_imagen = Path(ruta).stem  

#1.1  Seteo constantes
MARGEN = 0.001  #distancia minima al borde
RADIO_MIN = 0.01 #porcentaje minimo del radio con respecto al tamaño de la imagen
RADIO_MAX = 0.03 #0.02porcentaje maximo del radio con respecto al tamaño de la imagen

#1.2  Seleccionar imagen y obtener sus propiedades
imagen = cv2.imread(ruta)
alto = imagen.shape[0]
ancho = imagen.shape[1]
lado = min(alto,ancho)  #tomaremos el lado mas pequeño de la imagen para calcular el tamaño de la mancha en forma de porcentaje

#Adaptar la oscuridad de la mancha a los distintos fondos
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)   # crea una copia de la foto en escala de grises

intentos = 0
MAX_INTENTOS = 5 #bucle para x imagenes

while intentos < MAX_INTENTOS:
    intentos = intentos + 1

    # 2. Elegir al azar como sera la mancha
    margen = int(lado * MARGEN)
    x = random.randint (margen, ancho - margen)
    y = random.randint (margen, alto - margen)
    radio_ancho = random.randint(int(lado * RADIO_MIN), int(lado * RADIO_MAX))       
    radio_alto = random.randint(int(radio_ancho * 0.5), radio_ancho)    
    tamano = max(radio_ancho, radio_alto) #elige el maximo entre el radio_anco y el radio_alto
    
    # Recortar un cuadrado de la foto en grises alrededor de la mancha
    zona = gris[max(0, y - tamano) : y + tamano, max(0, x - tamano) : x + tamano]
    brillo_fondo = zona.mean()   # lo claro que es el fondo donde va la mancha (0 = negro, 255 = blanco)
    print("brillo del fondo:", round(brillo_fondo))
    objetivo = random.uniform(70, 120)   # nivel de gris al que queremos que llegue el centro de la mancha
    oscuridad = 1 - objetivo / brillo_fondo   # cuánto hay que oscurecer para pasar del fondo al objetivo
    print("brillo:", round(brillo_fondo), "- objetivo:", round(objetivo), "- oscuridad:", round(oscuridad, 2))
    oscuridad = min(max(oscuridad, 0.2), 0.85)   # limita la oscuridad entre 0.2 y 0.85

    
    giro = random.randint(0, 180)             # inclinación en grados
    #oscuridad = random.uniform(0.2, 0.5)      # cuánto oscurece

    # 3. Dibujar un círculo blanco en una imagen negra (el "molde")
    molde = np.zeros(imagen.shape[:2], dtype=np.float32)
    cv2.ellipse(molde, (x,y), (radio_ancho,radio_alto),giro,0,360,1.0,-1) #1.0 color blanco. 0 indica aqui no hay mancha, 1 indica aui si hay. El color blanco ees 255. Dibujamos la elipse de color blanco sobre el molde negro que creamos

    # 4. Difuminar el borde
    molde = cv2.GaussianBlur(molde, (0, 0), max(1,tamano/4)) #Mientras mayor sea el divisor, la difuminacion es menor.  Al hacer manchas pequenas tuvimos que setear un umbral porque si no, no se difuminan

    # 5. Oscurecer la foto donde está el molde
    resultado = imagen * (1 - 0.3 * molde[..., None]) #mientras mas chico el porcentaje, mas clara la mancha

    # 6. Textura: zonas más claras y más oscuras dentro de la mancha
    ruido = np.random.rand(alto, ancho).astype(np.float32)
    ruido = cv2.GaussianBlur(ruido, (0, 0), 1)
    ruido = (ruido - ruido.min()) / (ruido.max() - ruido.min())
    textura = 0.5 + 0.5 * ruido #la textura reduce el contraste/comprime al ruido. Si tenemos un pixel negro, uno gris y uno blanco, el ruido va de 0 a 1. Con la textura hacemos que vaya del 0.5 al 1

    # 7. Decidir cuanto oscurecer cada pixel. Esto solo actuara donde hay mancha. Porque si el pixel es 0, la multiplicacion dara 0 
    oscuridad_por_pixel = oscuridad * molde * textura
    resultado = imagen * (1 - oscuridad_por_pixel[..., None]) #el 1-opp le da el brillo. Se queda con el 1-opp de su brillo, por lo tanto se oscurece een ese opp. Ej 30%. Le deja un 70% de brillo y por eso se oscurece un 305

    # 8. Guardar imagenes. Crear el directorio
    cv2.imwrite(f"resultados/M9317/{nombre_imagen}_mancha_{intentos}.png", resultado.astype(np.uint8))
    cv2.imwrite(f"resultados/M9317/{nombre_imagen}_mascara_{intentos}.png", (molde * 255).astype(np.uint8))

    if molde.max() == 0:
        if molde.max() == 0:
            with open("resultados/mascaras_vacias.txt", "a") as log:
             log.write(f"{nombre_imagen}, imagen {i}, centro=({x},{y}), "
                       f"radios=({radio_ancho},{radio_alto}), giro={giro}\n")
            print("Aviso: máscara vacía, apuntada en mascaras_vacias.txt")
        break
