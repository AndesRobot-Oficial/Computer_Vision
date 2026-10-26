# Computer_Vision

Repositorio del equipo de Visión Computacional para RoboCup@Home. Aquí se desarrollan los módulos de percepción del robot: detección de personas, reconocimiento de identidad, estimación de postura, gestos y detección de objetos.

> **Estado general:** los módulos funcionan como prototipos en Python. La integración con **ROS 2** todavía está pendiente.

---

## 1. Objetivo del equipo

Al finalizar el proyecto, Visión debe ofrecer al resto del sistema las siguientes interfaces ROS 2.

### Interfaces

| Interfaz | Tipo | Dirección | Descripción |
|---|---|---|---|
| `/vision/persons` | Tópico | Publica | Personas detectadas y su posición |
| `/vision/faces` | Tópico | Publica | Identidad de los rostros reconocidos |
| `/vision/gestures` | Tópico | Publica | Gestos detectados: saludo, señalar, etc. |
| `/vision/identify_person` | Servicio | Ofrece | ¿Quién es esta persona? |
| `/vision/describe_person` | Servicio | Ofrece | Describir un atributo visual: ropa, color |
| `/vision/detect_bag` | Servicio | Ofrece | Ubicar la bolsa para la entrega |

### Nodos

| Nodo | Función prevista |
|---|---|
| `person_detector_node` | Detección y seguimiento de personas |
| `face_recognition_node` | Reconocimiento facial |
| `gesture_node` | Detección de gestos |
| `object_detector_node` | Detección de objetos |

---

## 2. Estado actual de los módulos

| Módulo | Estado | ROS 2 | Documentación |
|---|---|---|---|
| Dataset de objetos | Mejorado, en reorganización | No aplica | [README del dataset](#) |
| Reconocimiento de personas (ID recognition) | Prototipo funcional en Python | Pendiente | [README del módulo](#) |
| Estimación de postura (de pie / sentado) | Prototipo en desarrollo | Pendiente | README pendiente |
| Detección de gestos | [Por definir] | Pendiente | — |
| Detección de bolsa (`/vision/detect_bag`) | [Por definir] | Pendiente | — |

---

## 3. Avances por módulo

### 3.1. Dataset de objetos

Se mejoró de forma importante la calidad de las anotaciones y se está alineando con el **Rulebook RoboCup@Home 2027** (borrador).

- La clase genérica `cutlery` se dividió en `spoon`, `fork` y `knife`, porque la prueba de *Pick and Place* exige distinguir cada cubierto.
- Se corrigieron cajas, se agregaron las que faltaban y se retiraron utensilios que no son cubiertos (espátulas, cucharones, etc.).
- Se desarrolló **RoboCup Annotator**, una herramienta local en Python para corregir anotaciones COCO JSON de forma rápida, con filtros de revisada, modificada, sospechosa y excluida.
- Resultado en Roboflow: de 668 anotaciones en `cutlery` se pasó a 851 anotaciones de cubiertos separadas por clase.
- Línea base actual: `corregido1_yolo11n` (YOLO11n, 640×640), con mAP50 de 0.841 en el modelo completo. Aún no se reentrena con las clases nuevas.

Detalles completos, criterios de anotación y pendientes en el **[README del dataset](#)**.

### 3.2. Reconocimiento de personas (ID recognition)

Prototipo independiente de ROS 2 que identifica a una persona combinando **prenda (color del torso)** y **rasgo facial**.

- **Detección y seguimiento:** YOLO11n detecta personas y ByteTrack asigna IDs temporales.
- **Color del torso:** se estima sobre una región del torso, sin segmentación precisa de la ropa.
- **Reconocimiento facial:** OpenCV YuNet localiza el rostro y SFace genera vectores de 128 dimensiones, comparados por similitud coseno (umbral por defecto: `0.45`).
- **Salidas locales:** `detecciones.json` y `personas_memoria.json`.

Limitaciones conocidas: los IDs de ByteTrack no son identidades permanentes, el umbral aún debe validarse experimentalmente y la posición se da en píxeles, no en metros.

Instalación, uso y detalles en el **[README del módulo](#)**.

### 3.3. Estimación de postura (de pie / sentado)

Módulo que determina si una persona está **de pie o sentada**.

- **Tecnología:** MediaPipe Pose + OpenCV (`deteccion_esqueleto.py`).
- **Método:** se calculan ángulos entre puntos clave (hombro, cadera, rodilla) y se comparan con rangos de referencia para clasificar la postura.
- **Estabilidad:** el resultado se entrega a partir de un buffer de lecturas, en lugar de clasificar cuadro a cuadro.
- **Siguiente mejora:** migrar a la cámara de profundidad **RealSense** para calcular el ángulo de cadera en 3D y resolver la ambigüedad cuando la persona está de frente o de espaldas.

Problemas pendientes: cuando la persona se encuadra muy cerca, los puntos hombro-cadera-rodilla no son visibles y la clasificación falla.

> Este módulo aún no tiene README propio ni está integrado con ROS 2.

---

## 4. Hoja de ruta

- [ ] Reentrenar el modelo de objetos con las clases `spoon`, `fork` y `knife` y comparar contra la línea base.
- [ ] Revisar las imágenes sin anotar del dataset antes de crear la nueva versión.
- [ ] Agregar fotos de cubiertos tomadas con la RealSense.
- [ ] Validar el reconocimiento de personas con distintas iluminaciones, distancias y umbrales.
- [ ] Escribir el README de estimación de postura.
- [ ] Integrar la estimación de postura con la cámara RealSense.
- [ ] Convertir los módulos en nodos `rclpy` y publicar los tópicos y servicios de la sección 1.
- [ ] Implementar `gesture_node` y el servicio `/vision/detect_bag`.
- [ ] Validar la ejecución completa en Ubuntu con el robot.

---

## 5. Estructura del repositorio

```text
[ajustar según el repositorio]
├── dataset/               # Dataset y herramienta de anotación
├── id_recognition/        # Detección de personas, color del torso y reconocimiento facial
├── pose_estimation/       # Clasificación de postura (de pie / sentado)
└── README.md
```

---

## 6. Privacidad de datos

Los vectores faciales y las fotografías de personas pueden ser **datos biométricos sensibles**. No deben subirse a este repositorio. Las carpetas de registros (`face_data/`, `detecciones.json`, `personas_memoria.json`) deben estar en el `.gitignore`.

---

## 7. Equipo

Equipo de Visión Computacional, RoboCup@Home: [nombres de los integrantes].
