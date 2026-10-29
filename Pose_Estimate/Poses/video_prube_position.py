import cv2
import math
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs

# --- Inicialización de MediaPipe Pose ---
mp_pose = mp.solutions.pose

# --- Variables globales ---
ANGULO_DE_PIE_MIN = 120    # >= a esto => de pie
ANGULO_SENTADO_MAX = 110   # <= a esto => sentado
                            # entre 100 y 140 => indeterminado / transición

VISIBILITY_MIN = 0.5       # umbral para considerar un landmark confiable
DEPTH_MIN_M = 0.2          # rango válido del sensor (metros)
DEPTH_MAX_M = 6.0

TAMANO_LOTE = 10           # cantidad de lecturas por lote antes de decidir


# --- Funciones ---

def especifics_points(body_part, results, width, height):
    """Devuelve las coordenadas (x, y) en píxeles de un landmark específico."""
    if results.pose_landmarks:
        landmark = results.pose_landmarks.landmark[body_part.value]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return x, y
    return None, None


def visibilidad_punto(body_part, results):
    """Devuelve el valor de visibility (0-1) de un landmark específico."""
    if results.pose_landmarks:
        return results.pose_landmarks.landmark[body_part.value].visibility
    return 0.0


def elegir_lado(results):
    """
    Compara la visibilidad promedio del lado derecho vs izquierdo
    (hombro, cadera, rodilla) y devuelve el trío de landmarks del lado
    con mejor visibilidad.
    """
    lado_derecho = (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                     mp_pose.PoseLandmark.RIGHT_HIP,
                     mp_pose.PoseLandmark.RIGHT_KNEE)
    lado_izquierdo = (mp_pose.PoseLandmark.LEFT_SHOULDER,
                       mp_pose.PoseLandmark.LEFT_HIP,
                       mp_pose.PoseLandmark.LEFT_KNEE)

    vis_derecha = sum(visibilidad_punto(p, results) for p in lado_derecho) / 3
    vis_izquierda = sum(visibilidad_punto(p, results) for p in lado_izquierdo) / 3

    if vis_derecha >= vis_izquierda:
        return lado_derecho, vis_derecha
    return lado_izquierdo, vis_izquierda


def punto_3d(pixel_xy, depth_frame, intrinsics):
    """
    Convierte un píxel (x, y) + la profundidad leída del sensor en un
    punto 3D real (X, Y, Z) en metros, usando los intrínsecos de la cámara.
    Si la lectura de profundidad no es válida, hace un pequeño muestreo
    en una ventana 3x3 alrededor del píxel para evitar huecos puntuales
    del sensor (bordes, ropa oscura, reflejos).
    """
    x, y = pixel_xy
    if x is None or y is None:
        return None

    h, w = depth_frame.get_height(), depth_frame.get_width()
    if not (0 <= x < w and 0 <= y < h):
        return None

    profundidad = depth_frame.get_distance(x, y)

    if profundidad <= 0:
        # muestreo 3x3 como respaldo ante huecos del sensor
        muestras = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                xi, yi = x + dx, y + dy
                if 0 <= xi < w and 0 <= yi < h:
                    d = depth_frame.get_distance(xi, yi)
                    if d > 0:
                        muestras.append(d)
        if not muestras:
            return None
        profundidad = sum(muestras) / len(muestras)

    if not (DEPTH_MIN_M <= profundidad <= DEPTH_MAX_M):
        return None

    punto = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], profundidad)
    return np.array(punto)  # [X, Y, Z] en metros


def calcular_angulo_3d(a, b, c):
    """
    Calcula el ángulo interno en el punto 'b', formado por los vectores
    b->a y b->c, en el espacio 3D real (metros), usando producto punto.
    Devuelve el ángulo en grados (0-180).
    """
    if a is None or b is None or c is None:
        return None

    v1 = a - b
    v2 = c - b

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return None

    cos_ang = np.dot(v1, v2) / (norm1 * norm2)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)  # evita errores de precisión fuera de [-1, 1]

    return math.degrees(math.acos(cos_ang))


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
    if None in first or None in second:
        return
    cv2.circle(image, first, 5, (255, 0, 0), -1)
    cv2.circle(image, second, 5, (255, 0, 0), -1)
    cv2.line(image, first, second, (255, 255, 255), 2)


def dibujar_texto(image, texto, posicion, color=(0, 255, 0)):
    """Escribe texto sobre la imagen (ej. la postura detectada)."""
    cv2.putText(image, texto, posicion, cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)


def punto_3d_confiable(landmark, px, results, depth_frame, intrinsics):
    """
    Igual que punto_3d, pero primero verifica la visibilidad individual
    de ESE landmark puntual. Si el propio punto tiene baja confianza
    (por ejemplo, cadera/rodilla fuera de encuadre y MediaPipe
    extrapolando su posición sobre el fondo), lo descartamos antes de
    siquiera consultar la profundidad — evita leer distancias de la
    pared o la mesa en vez del cuerpo.
    """
    if visibilidad_punto(landmark, results) < VISIBILITY_MIN:
        return None
    return punto_3d(px, depth_frame, intrinsics)


def frame_detect(color_image, depth_frame, intrinsics, pose, lote_angulos, ultimo_resultado):
    """
    Procesa un frame de color: detecta landmarks con MediaPipe, elige el
    lado con mejor visibilidad, obtiene sus puntos 3D reales a partir del
    depth_frame y calcula el ángulo en 3D.

    Trabaja por LOTES: acumula ángulos válidos en 'lote_angulos' hasta
    juntar TAMANO_LOTE; en ese momento calcula el promedio, clasifica la
    postura, actualiza 'ultimo_resultado' (dict mutable, así el cambio
    se ve reflejado afuera de la función) y vacía el lote para arrancar
    a juntar el siguiente. Mientras se junta un lote nuevo, se sigue
    mostrando el último resultado ya determinado.

    Devuelve (imagen_con_lineas, mensaje, color_mensaje, mensaje_extra).
    """
    height, width, _ = color_image.shape
    image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    mensaje_extra = None

    if results.pose_landmarks is not None:
        (hombro_lm, cadera_lm, rodilla_lm), vis_promedio = elegir_lado(results)

        px_hombro = especifics_points(hombro_lm, results, width, height)
        px_cadera = especifics_points(cadera_lm, results, width, height)
        px_rodilla = especifics_points(rodilla_lm, results, width, height)

        hombro_3d = punto_3d_confiable(hombro_lm, px_hombro, results, depth_frame, intrinsics)
        cadera_3d = punto_3d_confiable(cadera_lm, px_cadera, results, depth_frame, intrinsics)
        rodilla_3d = punto_3d_confiable(rodilla_lm, px_rodilla, results, depth_frame, intrinsics)

        angulo_cadera = calcular_angulo_3d(hombro_3d, cadera_3d, rodilla_3d)

        points_line(px_hombro, px_cadera, color_image)
        points_line(px_cadera, px_rodilla, color_image)

        if angulo_cadera is not None:
            lote_angulos.append(angulo_cadera)
            lado = "derecho" if hombro_lm.name.startswith("RIGHT") else "izquierdo"
            print(f"Ángulo cadera (3D real): {angulo_cadera:.1f}° "
                  f"(lado: {lado}, visibilidad: {vis_promedio:.2f}, "
                  f"lote: {len(lote_angulos)}/{TAMANO_LOTE})")

        if len(lote_angulos) >= TAMANO_LOTE:
            angulo_promedio = sum(lote_angulos) / len(lote_angulos)
            postura = clasificar_postura(angulo_promedio)
            ultimo_resultado["texto"] = f"{postura} ({angulo_promedio:.1f} grados)"
            ultimo_resultado["color"] = (0, 255, 0)
            print(f"--- Lote completo -> veredicto: {postura} "
                  f"(promedio: {angulo_promedio:.1f}°) ---")
            lote_angulos.clear()  # arranca el siguiente lote desde cero

        if hombro_3d is None or cadera_3d is None or rodilla_3d is None:
            mensaje_extra = "Encuadra hombro, cadera y rodilla completos"
    else:
        lote_angulos.clear()  # se perdió a la persona: descartamos el lote a medias

    mensaje = f'{ultimo_resultado["texto"]}  [{len(lote_angulos)}/{TAMANO_LOTE}]'
    return color_image, mensaje, ultimo_resultado["color"], mensaje_extra


# --- Main: captura en vivo con RealSense (color + profundidad alineados) ---

if __name__ == "__main__":
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # Alinea el frame de profundidad al de color para que ambos compartan
    # la misma grilla de píxeles (así el (x, y) de MediaPipe sirve para
    # consultar directamente la profundidad).
    align = rs.align(rs.stream.color)

    lote_angulos = []
    ultimo_resultado = {"texto": "Recolectando datos...", "color": (0, 165, 255)}

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()

                # Procesamos SIN voltear: el depth_frame no está volteado,
                # así que los píxeles (x, y) que da MediaPipe siguen
                # correspondiendo exactamente a la misma posición en el
                # mapa de profundidad. Las líneas/puntos se dibujan acá
                # (son simétricos, no importa que se volteen después).
                frame_procesado, mensaje, color_mensaje, mensaje_extra = frame_detect(
                    color_image, depth_frame, intrinsics, pose, lote_angulos, ultimo_resultado
                )

                # Volteamos recién ahora, para mostrar en espejo...
                frame_procesado = cv2.flip(frame_procesado, 1)

                # ...y el texto se escribe DESPUÉS del flip, así no queda
                # espejado ni al revés.
                dibujar_texto(frame_procesado, mensaje, (30, 50), color_mensaje)
                if mensaje_extra:
                    dibujar_texto(frame_procesado, mensaje_extra, (30, 80), (0, 165, 255))

                cv2.imshow("Deteccion de Postura - RealSense (3D real)", frame_procesado)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            pipeline.stop()
            cv2.destroyAllWindows()