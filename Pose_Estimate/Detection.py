import cv2
import mediapipe as mp

# --- Inicialización de MediaPipe Pose ---
mp_pose = mp.solutions.pose #guardar la solución de pose
mp_drawing = mp.solutions.drawing_utils #dibujar la pose para visualizarla
mp_drawing_styles = mp.solutions.drawing_styles

# Configuración del modelo de pose
pose = mp_pose.Pose(
    static_image_mode=False,       # False = video en tiempo real
    model_complexity=1,            # 0 (rápido) - 2 (más preciso) es para la complejidad del modelo
    smooth_landmarks=True,          # Suaviza los puntos del cuerpo detectados
    min_detection_confidence=0.5,   # Confianza mínima para la detección de pose
    min_tracking_confidence=0.5     # Confianza mínima para el seguimiento de pose
)

# --- Inicialización de la cámara ---
# 0 = cámara por defecto. Cambia el índice si tienes varias cámaras.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: no se pudo acceder a la cámara.")
    exit()

print("Cámara iniciada. Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame de la cámara.")
        break

    # Voltear horizontalmente (efecto espejo, más natural)
    frame = cv2.flip(frame, 1)

    # MediaPipe trabaja en RGB, OpenCV captura en BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar el frame para detectar el esqueleto
    resultados = pose.process(rgb_frame)

    # Si se detectaron landmarks (puntos del cuerpo), dibujarlos
    if resultados.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            resultados.pose_landmarks, #puntos del cuerpo detectados
            mp_pose.POSE_CONNECTIONS, #lineas de conecciíon entre los puntos del cuerpo
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    # Mostrar el resultado en una ventana
    cv2.imshow("Deteccion de Esqueleto - MediaPipe", frame)

    #Imprimir los puntos de referencia detectados en la consola
    print("Puntos de referencia detectados:", resultados.pose_landmarks)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
pose.close()