# RoboCup Vision — Prototipo en Python para Windows y Ubuntu

Prototipo independiente de ROS 2 para detección de personas, seguimiento temporal, estimación del color del torso y reconocimiento facial experimental.

Actualmente, el proyecto está organizado en dos módulos de lógica en Python que todavía no son nodos ROS 2.

## 1. Arquitectura del prototipo

| Archivo | Funcionalidad |
|---|---|
| `person_detector_node.py` | YOLO detecta personas, ByteTrack asigna IDs temporales y estima el color del torso. |
| `face_recognition_node.py` | OpenCV YuNet localiza rostros y SFace genera vectores de características para intentar reconocer personas registradas. |
| `demo.py` | Integra ambos módulos, utiliza la cámara web y guarda archivos JSON. |

### Tecnologías utilizadas

- **YOLO11n:** detección de personas.
- **ByteTrack:** seguimiento de personas mediante IDs temporales.
- **OpenCV YuNet:** detección y localización facial.
- **OpenCV SFace:** extracción de características faciales y comparación mediante similitud coseno.
- **Python:** implementación del prototipo.
- **JSON y NumPy:** almacenamiento de resultados y vectores faciales.

## 2. Instalación en Windows 10/11

Se recomienda utilizar **Python 3.12 de 64 bits**.

### 2.1. Verificar Python

Abre PowerShell y ejecuta:

```powershell
py -3.12 --version
```

Si el comando muestra la versión instalada, continúa con el siguiente paso.

### 2.2. Preparar el entorno virtual

Abre PowerShell dentro de la carpeta `robocup_vision_demo`, donde se encuentra `demo.py`, y ejecuta:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2.3. Ejecutar el prototipo

```powershell
.\.venv\Scripts\python.exe demo.py --camera 0
```

**Nota:** no es necesario activar el entorno virtual. Por lo tanto, no hace falta modificar la política de ejecución de PowerShell.

Si `py -3.12` no está disponible, pero `python` apunta a una versión compatible, puedes sustituir `py -3.12` por `python` en los comandos anteriores.

### 2.4. Solución de problemas con la cámara

Si la cámara no se abre:

1. Cierra aplicaciones que puedan estar utilizando la cámara, como Zoom o Microsoft Teams.
2. Prueba con otro índice de cámara:

   ```powershell
   .\.venv\Scripts\python.exe demo.py --camera 1
   ```

3. Revisa los permisos en **Configuración → Privacidad y seguridad → Cámara**.
4. Habilita el acceso a la cámara para las aplicaciones de escritorio.

## 3. Instalación en Ubuntu

Los siguientes pasos corresponden a la instalación prevista para Ubuntu.

### 3.1. Instalar dependencias del sistema

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip libgl1 libglib2.0-0
```

### 3.2. Crear el entorno virtual

Desde la carpeta del proyecto, ejecuta:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3. Ejecutar el prototipo

```bash
python demo.py --camera 0
```

**Importante:** en la primera ejecución, el sistema descargará automáticamente los modelos necesarios:

- `yolo11n.pt`, para la detección de personas.
- Modelos ONNX oficiales de OpenCV Zoo para YuNet y SFace.

Se requiere conexión a internet para descargar los modelos que todavía no estén disponibles localmente.

## 4. Uso del prototipo

### 4.1. Preparación

1. Ejecuta `demo.py`.
2. Colócate frente a la cámara.
3. Procura que tu rostro y torso sean visibles.
4. Utiliza los controles del teclado para interactuar con el sistema.

### 4.2. Controles

| Tecla | Acción |
|---|---|
| `TAB` | Selecciona una persona, identificada mediante un recuadro amarillo. |
| `E` | Registra una muestra facial de la persona seleccionada, con su consentimiento. |
| `Q` o `ESC` | Cierra la aplicación. |

Para registrar una persona, pulsa `E` varias veces mientras la persona seleccionada aparece en distintas tomas. Esto permite obtener varias muestras para el registro facial.

### 4.3. Prueba de seguimiento y reconocimiento

1. Permite que el sistema detecte a una persona.
2. Selecciona a la persona con `TAB`.
3. Registra su rostro con `E`.
4. Sal del campo visual de la cámara.
5. Regresa frente a ella y observa los resultados.

El identificador de ByteTrack puede cambiar cuando la persona desaparece y vuelve a entrar en escena. El reconocimiento facial intenta recuperar la identidad registrada y asociarla con un identificador persistente.

## 5. Resultados generados

El prototipo genera archivos de resultados y datos locales.

| Archivo o carpeta | Descripción |
|---|---|
| `face_data/id_N/*.npy` | Vectores faciales registrados mediante SFace. |
| `face_data/id_N/*.png` | Fotografías del sistema anterior basado en LBPH, si todavía existen. No se reutilizan en el nuevo sistema SFace. |
| `detecciones.json` | Instantánea del último procesamiento, con `track_id`, `person_id`, caja de detección, posición central en píxeles y color estimado. |
| `personas_memoria.json` | Color más frecuente observado para cada identidad durante la sesión. |
| `models/` | Modelos ONNX de YuNet y SFace descargados para el reconocimiento facial. |

### Consideraciones sobre los datos

Los vectores faciales y las fotografías pueden constituir **datos biométricos sensibles**. No deben subirse al repositorio público de GitHub.

Se recomienda mantener los datos generados fuera del control de versiones mediante un archivo `.gitignore`:

```gitignore
.venv/
__pycache__/
*.py[cod]

face_data/
detecciones.json
personas_memoria.json

# Los modelos descargados pueden excluirse si no se versionan.
# models/
```

Si el proyecto requiere distribuir los modelos, puede gestionarse la carpeta `models/` por separado, según las necesidades del equipo.

## 6. Actualización: reconocimiento facial con SFace

El módulo `face_recognition_node.py` fue actualizado para sustituir el enfoque anterior basado en Haar y LBPH por un sistema basado en **YuNet y SFace**.

### 6.1. Flujo de reconocimiento

1. **Detección facial:** YuNet localiza el rostro en la imagen.
2. **Extracción de características:** SFace transforma el rostro en un vector numérico de 128 dimensiones.
3. **Registro:** al pulsar `E`, se guardan los vectores en archivos `.npy` dentro de `face_data/id_N/`.
4. **Comparación:** cuando una persona vuelve a aparecer, el sistema compara los vectores mediante similitud coseno.
5. **Asignación de identidad:** el sistema intenta asociar el rostro detectado con una identidad previamente registrada.

### 6.2. Umbral de similitud

El umbral predeterminado es `0.45`.

Puede ajustarse mediante el argumento `--threshold`. Por ejemplo:

```powershell
.\.venv\Scripts\python.exe demo.py --camera 0 --threshold 0.50
```

Un umbral más alto exige una similitud mayor para aceptar una coincidencia y puede reducir algunos falsos positivos, pero también puede aumentar los falsos negativos.

El valor adecuado debe determinarse mediante pruebas con personas, condiciones de iluminación y distancias variadas.

### 6.3. Descarga de modelos

En la primera ejecución, el sistema descarga los dos modelos ONNX oficiales de OpenCV Zoo en la carpeta `models/`.

Se requiere conexión a internet para esta descarga inicial, salvo que los archivos ya estén disponibles localmente.

### 6.4. Compatibilidad con el registro anterior

Las fotografías `.png` generadas por el sistema anterior basado en LBPH **no se reutilizan** para el reconocimiento con SFace.

Por lo tanto, es necesario registrar nuevamente los rostros utilizando la versión actualizada del prototipo.

## 7. Limitaciones conocidas

- **Seguimiento temporal:** los IDs de ByteTrack no representan identidades permanentes.
- **Reconocimiento facial experimental:** SFace puede confundir personas o no reconocerlas debido a cambios de iluminación, pose, distancia u oclusiones.
- **Umbral heurístico:** el valor predeterminado de `0.45` debe validarse experimentalmente para las condiciones de uso.
- **Estimación de color:** el color se calcula sobre una región del torso y no mediante una segmentación precisa de la ropa.
- **Posición 2D:** la ubicación se expresa en píxeles de imagen, no en coordenadas del mundo real ni en metros.
- **Integración con ROS 2:** los módulos son programas de Python; todavía no son nodos `rclpy` ni implementan los tópicos y servicios definidos en el diagrama del sistema.
- **Validación física:** el paquete no ha sido probado físicamente con una webcam en Windows. Debe tratarse como una base experimental y ajustarse según los resultados obtenidos.
- **Privacidad:** el registro y la comparación facial requieren consentimiento y medidas adecuadas para proteger los datos biométricos.

## 8. Próximos pasos

- [ ] Validar la detección de personas y el seguimiento con diferentes condiciones de iluminación.
- [ ] Evaluar el reconocimiento facial con distintos umbrales de similitud.
- [ ] Medir falsos positivos y falsos negativos del reconocimiento.
- [ ] Verificar la estabilidad de la estimación del color del torso.
- [ ] Validar la ejecución en Ubuntu.
- [ ] Integrar los módulos como nodos ROS 2 mediante `rclpy`.
- [ ] Implementar los tópicos y servicios necesarios para la integración con el sistema RoboCup.
- [ ] Proteger los datos biométricos y excluir los registros personales del repositorio.

---

**Estado del proyecto:** prototipo experimental en Python, con detección de personas, seguimiento temporal y reconocimiento facial mediante YuNet y SFace. La integración completa con ROS 2 y su validación física siguen pendientes.
