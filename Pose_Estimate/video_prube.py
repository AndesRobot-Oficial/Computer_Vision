import cv2
import math
import mediapipe as mp

# --- Inicialización de MediaPipe Pose ---
mp_pose = mp.solutions.pose        # guardar la solución de pose

# --- Variables globales ---
ANGULO_DE_PIE_MIN = 140    # >= a esto => de pie
ANGULO_SENTADO_MAX = 100   # <= a esto => sentado
                            # entre 120 y 160 => indeterminado / transición


# --- Funciones ---

def especifics_points(body_part, results, width, height):
    """Devuelve las coordenadas (x, y) en píxeles de un landmark específico."""
    if results.pose_landmarks:
        landmark = results.pose_landmarks.landmark[body_part.value]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return x, y
    return None, None


def calcular_angulo(a, b, c):
    """
    Calcula el ángulo interno en el punto 'b', formado por los segmentos
    b->a y b->c, usando atan2 (seno y coseno) sobre los vectores.
    Devuelve el ángulo en grados (0-180).
    """
    if None in a or None in b or None in c:
        return None

    ang_a = math.atan2(a[1] - b[1], a[0] - b[0])
    ang_c = math.atan2(c[1] - b[1], c[0] - b[0])
    angulo = math.degrees(ang_a - ang_c)
    angulo = abs(angulo)

    if angulo > 180:
        angulo = 360 - angulo

    return angulo


def clasificar_postura(angulo):
    """Clasifica la postura según el ángulo de cadera calculado."""
    if angulo is None:
        return "Sin datos"
    elif angulo >= ANGULO_DE_PIE_MIN:
        return "De pie"
    elif angulo <= ANGULO_SENTADO_MAX:
        return "Sentado"
    else:
        return "Indeterminado"


def points_line(first, second, image):
    """Dibuja dos puntos y la línea que los conecta."""
    cv2.circle(image, first, 5, (255, 0, 0), -1)
    cv2.circle(image, second, 5, (255, 0, 0), -1)
    cv2.line(image, first, second, (255, 255, 255), 2)


def dibujar_texto(image, texto, posicion):
    """Escribe texto sobre la imagen (ej. la postura detectada)."""
    cv2.putText(image, texto, posicion, cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)


def image_detect(image):
    
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)

    if results.pose_landmarks is not None:
        hombro = especifics_points(mp_pose.PoseLandmark.RIGHT_SHOULDER, results, width, height)
        cadera = especifics_points(mp_pose.PoseLandmark.RIGHT_HIP, results, width, height)
        rodilla = especifics_points(mp_pose.PoseLandmark.RIGHT_KNEE, results, width, height)

        angulo_cadera = calcular_angulo(hombro, cadera, rodilla)
        postura = clasificar_postura(angulo_cadera)

        print(f"Ángulo cadera: {angulo_cadera:.1f}° -> {postura}")

        points_line(hombro, cadera, image)
        points_line(cadera, rodilla, image)
        dibujar_texto(image, f"{postura} ({angulo_cadera:.1f} grados)", (30, 50))

# --- Main ---

if __name__ == "__main__":
    with mp_pose.Pose(static_image_mode=True) as pose:

        img2 = cv2.imread("Poses/sentado.jpg") 
        img1 = cv2.imread("Poses/depie.jpg")
        img3 = cv2.imread("Poses/sentado2.jpg")
        img4 = cv2.imread("Poses/depie2.jpg")

        img1 = cv2.resize(img1, (300, 300))
        img2 = cv2.resize(img2, (300, 300))
        img3 = cv2.resize(img3, (300, 300))
        img4 = cv2.resize(img4, (300, 300))

        image_detect(img1)
        image_detect(img2)
        image_detect(img3)
        image_detect(img4)

        fila_superior = cv2.hconcat([img1, img2])
        fila_inferior = cv2.hconcat([img3, img4])


        box = cv2.vconcat([fila_superior, fila_inferior])
        
        cv2.imshow("Detección de Postura - MediaPipe",box )
        cv2.waitKey(0)


    # --- Cierre ---
    cv2.destroyAllWindows()