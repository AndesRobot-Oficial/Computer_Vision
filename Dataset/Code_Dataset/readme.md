# ROS2 RealSense + YOLO Object Detection Node

This ROS 2 package performs real-time object detection using an Intel RealSense D435i camera and a YOLO model (ONNX format).  
It publishes object labels, 3D coordinates, and a binary detection flag.

---

# 📌 Features

- Real-time YOLO inference
- Depth-based 3D coordinate estimation
- Confidence filtering (75%)
- Detection flag publishing (1 / 0)
- OpenCV visualization window

---

# 📡 Published Topics

| Topic | Type | Description |
|--------|------|------------|
| `/yolo_detection_flag` | `std_msgs/Int32` | 1 if object detected, 0 otherwise |
| `/detected_object` | `std_msgs/String` | Detected object label |
| `/object_position` | `geometry_msgs/Point` | 3D position (X, Y, Z in meters) |

---

# ⚙️ System Requirements

- Ubuntu 22.04 (Recommended)
- ROS2 Humble Hawksbill
- Intel RealSense D435i
- Python 3.10+

---

# 🛠 Installation Guide

## 1️⃣ Install ROS2 Humble

```bash
sudo apt update
sudo apt install ros-humble-desktop
```

Source ROS2:

```bash
source /opt/ros/humble/setup.bash
```

---

## 2️⃣ Install ROS Dependencies

```bash
sudo apt install python3-colcon-common-extensions
sudo apt install ros-humble-geometry-msgs
sudo apt install ros-humble-std-msgs
sudo apt install ros-humble-sensor-msgs
```

---

## 3️⃣ Install Python Dependencies

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install pyrealsense2
```

---

## 4️⃣ Install Intel RealSense SDK

```bash
sudo apt install librealsense2-dev
sudo apt install librealsense2-utils
```

Test the camera:

```bash
realsense-viewer
```

---

# 🏗 Workspace Setup

Create a ROS2 workspace:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

Clone or copy this package into `src/`.

Build the workspace:

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

---

# ▶️ Running the Node

```bash
ros2 run Computer_vision detect
```

If using a custom model path:

```bash
ros2 run Computer_vision detect --ros-args -p model_path:=Models/best.onnx
```

---

# 🧠 Detection Logic

- Confidence threshold: 0.75
- If at least one object is detected → publishes `1`
- If no object is detected → publishes `0`

---

# 🖥 Visualization

An OpenCV window will open displaying:

- Bounding boxes
- Confidence score
- Estimated depth (meters)

Exit using `Ctrl+C`.

---

# 📂 Package Structure

```
Computer_vision/
├── Computer_vision/
│   ├── detect.py
│   ├── Models/
│   │   └── best.onnx
│   └── __init__.py
├── package.xml
├── setup.py
├── setup.cfg
```

---

# ⚠️ Notes

- GPU acceleration requires CUDA installed.
- If running on CPU only, performance will decrease.
- Ensure the camera is connected before starting the node.
- Always run:

```bash
source install/setup.bash
```

after building.

---

# 🛠 Troubleshooting

Check connected devices:

```bash
lsusb
```

List active topics:

```bash
ros2 topic list
```

Check detection flag:

```bash
ros2 topic echo /yolo_detection_flag
```

---

---

# 🇪🇸 Versión en Español

---

# Nodo ROS2 RealSense + YOLO

Este paquete ROS 2 realiza detección de objetos en tiempo real utilizando una cámara Intel RealSense D435i y un modelo YOLO en formato ONNX.

Publica etiquetas de objetos, coordenadas 3D y una bandera binaria de detección.

---

# 📌 Características

- Inferencia YOLO en tiempo real
- Estimación de coordenadas 3D con profundidad
- Filtro de confianza (75%)
- Publicación de bandera de detección (1 / 0)
- Visualización con OpenCV

---

# 📡 Tópicos Publicados

| Tópico | Tipo | Descripción |
|--------|------|------------|
| `/yolo_detection_flag` | `std_msgs/Int32` | 1 si detecta, 0 si no |
| `/detected_object` | `std_msgs/String` | Nombre del objeto detectado |
| `/object_position` | `geometry_msgs/Point` | Coordenadas 3D (X, Y, Z en metros) |

---

# ⚙️ Requisitos

- Ubuntu 22.04
- ROS2 Humble
- Intel RealSense D435i
- Python 3.10+

---

# 🛠 Instalación

Instalar ROS2:

```bash
sudo apt update
sudo apt install ros-humble-desktop
```

Dependencias ROS:

```bash
sudo apt install ros-humble-geometry-msgs
sudo apt install ros-humble-std-msgs
sudo apt install ros-humble-sensor-msgs
```

Dependencias Python:

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install pyrealsense2
```

Instalar RealSense SDK:

```bash
sudo apt install librealsense2-dev
sudo apt install librealsense2-utils
```

---

# 🏗 Compilación

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

---

# ▶️ Ejecutar

```bash
ros2 run Computer_vision detect
```

Con modelo personalizado:

```bash
ros2 run Computer_vision detect --ros-args -p model_path:=Models/best.onnx
```

---

# 🧠 Lógica de Detección

- Umbral de confianza: 0.75
- Si detecta al menos un objeto → publica `1`
- Si no detecta → publica `0`

---

# 🖥 Visualización

Se abre una ventana mostrando:

- Bounding boxes
- Confianza
- Distancia estimada

Salir con `Ctrl+C`.

---

# ⚠️ Notas

- Para usar GPU se requiere CUDA.
- Verificar que la cámara esté conectada.
- Ejecutar siempre:

```bash
source install/setup.bash
```

después de compilar.

---
