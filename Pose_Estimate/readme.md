# 🧍‍♂️ Estimación de postura: sentado / de pie

Conjunto de 4 programas desarrollados por el equipo de Visión Computacional de RoboCup @Home para determinar si una persona está **sentada** o **de pie**, usando **MediaPipe Pose** y, en la versión final, una cámara **Intel RealSense** para calcular el ángulo en 3D real.

> ⚠️ **Estado:** los programas `video_prube.py` y `video_prube_position.py` funcionan correctamente. La integración con **ROS 2 Humble** está pendiente.

---

## 📑 Contenido

| # | Archivo | Tipo | Entrada | ¿Clasifica postura? |
|---|---------|------|---------|---------------------|
| 1 | `detection.py` | Prototipo | Webcam | ❌ Solo dibuja el esqueleto |
| 2 | `image_prube` | Prueba | Imagen (`image.jpg`) | ❌ Solo marca puntos de rodillas |
| 3 | `video_prube.py` | Funcional | Imágenes en `Poses/` | ✅ Ángulo 2D |
| 4 | `video_prube_position.py` | **Versión final** | RealSense (color + profundidad) | ✅ Ángulo 3D + lotes |

---

## 🧰 Requisitos

- Python 3.9 – 3.11 (recomendado 3.10, la versión de Ubuntu 22.04 / ROS 2 Humble)
- Cámara web (programa 1)
- Cámara **Intel RealSense** (D435 / D435i o similar) y el SDK librealsense (programa 4)

### Librerías

| Librería | Usada en | Para qué |
|----------|----------|----------|
| `opencv-python` | 1, 2, 3, 4 | Captura, dibujo y visualización |
| `mediapipe` | 1, 2, 3, 4 | Estimación de pose (landmarks del cuerpo) |
| `numpy` | 4 | Vectores y producto punto en 3D |
| `pyrealsense2` | 4 | Acceso a la RealSense y deproyección a 3D |

### Instalación

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install "numpy<2" opencv-python mediapipe==0.10.14 pyrealsense2
```

> Se fija `mediapipe==0.10.14` porque los programas usan la API clásica `mp.solutions.pose`. Si hay conflictos con versiones nuevas de NumPy, usa `numpy<2`.
> El programa 2 importa `matplotlib` (`from matplotlib import image`), pero no se usa; esa línea se puede borrar.

---

## 🚀 Descripción de cada programa

### 1. `detection.py` — Prototipo de MediaPipe

Prueba inicial de la librería. Abre la webcam, detecta el esqueleto con `mp.solutions.pose` y lo dibuja en tiempo real con `drawing_utils`. También imprime los landmarks en consola.

```bash
python detection.py
```

- Salir: tecla `q`.
- Parámetros ajustables en `mp_pose.Pose(...)`: `model_complexity` (0 rápido – 2 preciso), `smooth_landmarks`, `min_detection_confidence`, `min_tracking_confidence`.
- Si tienes varias cámaras, cambia el índice en `cv2.VideoCapture(0)`.

### 2. `image_prube` — Prueba con imágenes

Primera prueba de estimación de pose sobre una imagen estática. Lee `image.jpg`, detecta las rodillas derecha e izquierda, las marca y las une con una línea.

```bash
python image_prube.py
```

- Requiere un archivo `image.jpg` en la misma carpeta.
- Funciones principales: `especifics_ponits()` (landmark → píxeles) y `points_line()` (dibuja puntos y línea).
- Cierra la ventana con cualquier tecla.

### 3. `video_prube.py` — Clasificación con ángulo 2D

Calcula el **ángulo de la cadera** (hombro → cadera → rodilla, lado derecho) en 2D con `atan2` y lo clasifica como *De pie*, *Sentado* o *Indeterminado*. Muestra el resultado sobre la imagen.

```bash
python video_prube.py
```

- En la versión actual procesa 4 imágenes de la carpeta `Poses/` (`depie.jpg`, `depie2.jpg`, `sentado.jpg`, `sentado2.jpg`) y las muestra en una cuadrícula de 2×2.
- Limitación: al ser 2D, la persona debe estar **de perfil**; de frente o de espaldas los puntos se superponen y el ángulo no es confiable. Para eso existe el programa 4.

### 4. `video_prube_position.py` — Versión final con RealSense (3D)

Es la versión más precisa. Usa la RealSense para obtener color y profundidad alineados, convierte los puntos hombro–cadera–rodilla a coordenadas 3D reales (metros) y calcula el ángulo con producto punto.

```bash
python video_prube_position.py
```

Funcionamiento:

1. **Alineación** de profundidad con color (`rs.align`), de modo que el `(x, y)` de MediaPipe sirve directamente para leer la profundidad.
2. **Selección de lado** (`elegir_lado`): usa el lado (derecho o izquierdo) con mayor visibilidad promedio.
3. **Filtro por visibilidad** (`punto_3d_confiable`): descarta landmarks poco confiables (por ejemplo, cadera o rodilla fuera de encuadre).
4. **Profundidad robusta** (`punto_3d`): si un píxel no tiene lectura, promedia una ventana 3×3 y valida el rango del sensor.
5. **Ángulo 3D** (`calcular_angulo_3d`): ángulo interno en la cadera.
6. **Decisión por lotes**: acumula `TAMANO_LOTE` lecturas válidas, promedia, clasifica y muestra el veredicto. Mientras llena el siguiente lote, conserva el último resultado.
7. La imagen se **voltea después** de procesar (efecto espejo) y el texto se dibuja tras el flip para que no salga invertido.

Salir: tecla `q`.

---

## ⚙️ Qué cambiar para ajustar la detección

### Umbrales de ángulo (los más importantes)

| Variable | `video_prube.py` | `video_prube_position.py` | Efecto |
|----------|:---:|:---:|--------|
| `ANGULO_DE_PIE_MIN` | 140 | 120 | Ángulo **≥** este valor → *De pie* |
| `ANGULO_SENTADO_MAX` | 100 | 110 | Ángulo **≤** este valor → *Sentado* |

Entre ambos valores el resultado es *Indeterminado* (zona de transición). Si los acercas, hay menos indeterminados pero más falsos positivos; si los alejas, el sistema es más conservador.

> Los umbrales **no son intercambiables** entre programas: el ángulo 2D (programa 3) y el 3D (programa 4) dan valores distintos para la misma postura. Recalibra si cambias de método.
> Nota: en el programa 4 los comentarios junto a las constantes mencionan otros rangos ("entre 100 y 140"); lo que manda son los valores asignados.

### Parámetros del programa 4 (`video_prube_position.py`)

| Variable | Valor por defecto | Qué controla |
|----------|:---:|--------------|
| `VISIBILITY_MIN` | 0.5 | Visibilidad mínima de un landmark para usarlo. Súbela si hay falsos positivos con partes del cuerpo fuera de cuadro |
| `DEPTH_MIN_M` / `DEPTH_MAX_M` | 0.2 / 6.0 | Rango válido de profundidad en metros |
| `TAMANO_LOTE` | 10 | Lecturas por lote antes de dar veredicto. Más alto = más estable pero más lento; más bajo = más reactivo |

### Funciones y clases clave

| Elemento | Archivo | Qué permite cambiar |
|----------|---------|---------------------|
| `clasificar_postura(angulo)` | 3 y 4 | Lógica y etiquetas de clasificación |
| `elegir_lado(results)` | 4 | Criterio para elegir lado (en el 3 está fijo el lado derecho) |
| `calcular_angulo()` / `calcular_angulo_3d()` | 3 / 4 | Método de cálculo del ángulo (2D con `atan2` / 3D con producto punto) |
| `mp_pose.Pose(...)` | 1, 2, 3, 4 | `model_complexity`, `min_detection_confidence`, `min_tracking_confidence`, `static_image_mode` |
| `config.enable_stream(...)` | 4 | Resolución y FPS de la RealSense (por defecto 640×480 a 30 FPS) |

---

## ✅ Qué tener en cuenta para un buen funcionamiento

- **Encuadre completo:** hombro, cadera y rodilla deben ser visibles. Si la persona está demasiado cerca, el programa 4 muestra *"Encuadra hombro, cadera y rodilla completos"*.
- **Distancia recomendada:** dentro del rango de profundidad configurado (0.2 – 6 m); para mejores resultados, de 1.5 a 3 m.
- **Iluminación:** buena y uniforme. La ropa muy oscura o superficies reflectantes pueden generar huecos en la profundidad.
- **Un solo sujeto:** MediaPipe Pose rastrea a una persona por frame.
- **Persona fuera de cuadro:** el programa 4 descarta el lote en curso y vuelve a empezar.
- **Programa 3:** pensado para personas de perfil por ser 2D.
- **Programa 4:** la RealSense debe estar conectada por **USB 3.0** y con el SDK instalado.
- Recalibra los umbrales si cambias la altura o el ángulo de montaje de la cámara (por ejemplo, al montarla en el robot).

---

## 🤖 Próximos pasos

- [ ] Integrar `video_prube_position.py` con **ROS 2 Humble** (nodo publicador de la postura).
- [ ] Validar los umbrales con más personas y posiciones.
- [ ] Manejar varias personas en escena.

---
