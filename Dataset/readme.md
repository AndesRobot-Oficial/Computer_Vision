# Registro de cambios — Dataset de cubiertos (RoboCup@Home)

**Responsable:** Piero Vilcapoma  
**Fecha:** 24-10-2026  
**Tarea:** Corrección de anotaciones de la clase `cutlery` y preparación del dataset para Roboflow.  
**Equipo:** Visión — dataset (Juan Diego, Valentina, Piero, Torres, Angelica, Manuel)

## 0. Contexto: por qué se dividió `cutlery`

El dataset se está reorganizando para alinearlo con el **Rulebook RoboCup@Home 2027** (borrador rev. 2026-08-31):

- **Sección 3.2.5 (p. 14):** la configuración mínima de objetos incluye los cubiertos **tenedor, cuchillo y cuchara** (*Cutlery: fork, knife, and spoon*).
- **Prueba 5.2 – Pick and Place (p. 40-41):** el robot debe servir un desayuno con **bowl, cuchara (spoon), leche y cereal**, y en la mesa a recoger hay *"one piece (fork, knife, or spoon)"*.

Con una sola clase `cutlery` el robot detecta "un cubierto", pero no puede saber cuál es la cuchara que pide la prueba. Por eso la clase se dividió en `spoon`, `fork` y `knife`. Estos nombres son además los que el equipo de **HRI** enviará como texto a Visión (por ejemplo `"spoon"`), así que deben escribirse exactamente así.

## 1. Estado inicial

Se asignó la revisión de **aproximadamente 500 imágenes** con la etiqueta genérica `cutlery` (**668 anotaciones** en Roboflow). Dentro de esta clase había objetos distintos, principalmente cucharas, tenedores y cuchillos. También se encontraron objetos sin una caja de detección y casos de utensilios ajenos a las categorías requeridas, como espátulas.

## 2. Cambios realizados

- Se reemplazó la etiqueta general `cutlery` por las clases específicas **`spoon`**, **`fork`** y **`knife`**, según correspondiera.
- Se corrigieron cajas existentes y se dibujaron nuevos *bounding boxes* para objetos que no estaban anotados.
- Se identificaron imágenes que debían **excluirse**, especialmente las de espátulas fuera de las clases objetivo, y se separaron de la exportación destinada a Roboflow.
- Se marcaron imágenes **sospechosas** para consultar posibles problemas con el equipo encargado del dataset.

### Criterios aplicados

| Objeto | Decisión |
|---|---|
| Cuchara de mesa, cucharita | `spoon` |
| Tenedor de mesa | `fork` |
| Cuchillo de mesa | `knife` |
| Espátula, cucharón, cuchara de palo, pinza, batidor | **No es cubierto**: se borra la caja; si es lo único en la foto, se excluye la imagen |
| Cuchillo grande de cocina | No es `knife` (solo cuchillo de mesa) |
| Una caja que cubría varios cubiertos juntos | Se reemplaza por **una caja por cubierto** |

**Regla corta:** *si se usa para comer, es cubierto; si se usa para cocinar, no.*

## 3. Resultados registrados

| Indicador | Cantidad |
|---|---:|
| Imágenes consideradas al inicio de la tarea | Aproximadamente **500** |
| Imágenes incluidas en la carga reportada por Roboflow | **372** |
| Imágenes que Roboflow identificó como duplicadas | **366** |
| Imágenes de esa carga no identificadas como duplicadas | **6** |

Roboflow informó que **aplicó las anotaciones enviadas** a las imágenes existentes detectadas como duplicadas. Estas cifras corresponden a esa carga y no deben interpretarse como un recuento definitivo de imágenes eliminadas o conservadas.

### Anotaciones en Roboflow antes y después (Classes & Tags, 23-10-2026)

| Clase | Antes | Después |
|---|---:|---:|
| `cutlery` | 668 | **6** |
| `spoon` | — | **185** |
| `fork` | — | **371** |
| `knife` | — | **289** |
| **Total cubiertos** | **668** | **851** |

El total subió en **183 cajas**: corresponden principalmente a cubiertos que antes no tenían caja o que estaban agrupados en una sola. Quedan **6 anotaciones** en `cutlery` por reasignar; cuando llegue a 0, la clase se puede eliminar.

**Balance:** `spoon` es la que tiene menos anotaciones (185) y es justamente la que pide la prueba del desayuno. Es la primera a reforzar.

## 4. Notebook y herramienta desarrollados

Se desarrolló **RoboCup Annotator**, una aplicación local en Python para acelerar las correcciones sin tener que editar cada anotación manualmente en Roboflow. Se ejecuta mediante el notebook **`Lanzador_RoboCup_Annotator.ipynb`**, que prepara las dependencias y abre la aplicación en una ventana independiente.

La herramienta permite cargar datasets **COCO JSON**, elegir clases de origen y destino, cambiar etiquetas, dibujar y ajustar cajas, visualizar polígonos y convertirlos en cajas, buscar imágenes y guardar el avance. También incorpora filtros y marcas de **revisada**, **modificada**, **sospechosa** y **excluida**.

Al exportar, organiza los archivos en carpetas separadas:

- `Roboflow/`: datos que se subirán.
- `Sospechosas/`: imágenes pendientes de consulta.
- `Exclusiones/`: imágenes apartadas, junto con sus motivos.

También permite generar un ZIP del dataset importable.

La conversión de polígonos a cajas también resuelve un problema detectado en el entrenamiento anterior: YOLO descartó imágenes por el aviso *"labels mix segment and detection rows"* (12 en train y 1 en validación).

## 5. Línea base: último modelo entrenado

**Todavía no se ha reentrenado con estos cambios.** El último modelo es **`corregido1_yolo11n`** (versión Roboflow v2 *Corregido1.0*, YOLO11n, 30 épocas, 640×640), entrenado **antes** de la división, por lo que solo conoce la clase `cutlery`. Sus resultados sirven como punto de comparación:

| Métrica de `cutlery` (validación, 146 instancias) | Valor |
|---|---:|
| Exactitud (diagonal de la matriz normalizada) | **0.68** |
| Precisión | 0.54 |
| Recall | 0.66 |
| mAP50 | 0.625 |
| mAP50-95 | 0.437 |
| Falsos positivos sobre fondo | **93** |
| Confundidos como `tool` | 6 % |

Modelo completo como referencia: mAP50 **0.841**, 10 de 19 clases con exactitud ≥ 0.80.

**Lectura:** los 93 falsos positivos y el 34 % de cubiertos perdidos como fondo eran consistentes con cajas faltantes y con objetos ajenos (espátulas) etiquetados como cubiertos, que es exactamente lo que se corrigió.

Además, el script de búsqueda de etiquetas faltantes (`buscar_sin_etiquetar.py`, 16-10-2026) encontró **55 imágenes** donde el modelo veía cubiertos sin caja, y 9 casos de posible confusión con `tool`. Esas imágenes estaban asignadas a esta tarea.

## 6. Qué se espera al reentrenar

- **Menos falsos positivos**, al quitar las espátulas y otros utensilios.
- **Más recall**, por las 183 cajas nuevas.
- **Ojo:** dividir en 3 clases es una tarea más difícil que detectar "cubierto". Cada clase tiene menos ejemplos que la clase original, y una cuchara y un tenedor vistos de lado se parecen. Es normal que la exactitud **por clase** arranque por debajo del 0.68 anterior, sobre todo en `spoon` (185). La comparación justa es la confusión entre `spoon`/`fork`/`knife` en la nueva matriz de confusión.

## 7. Pendientes

| # | Pendiente | Responsable |
|---|---|---|
| 1 | Reasignar las **6** anotaciones restantes de `cutlery` y eliminar la clase | Piero |
| 2 | Resolver con el equipo las imágenes de **`Sospechosas/`** | Piero + Juan Diego |
| 3 | Subir `spoon`, `fork` y `knife` hasta **~600 anotaciones cada una**, empezando por `spoon` | Piero |
| 4 | Agregar fotos tomadas con la **RealSense** de cubiertos sobre la mesa, junto a platos y bowls | Piero (Angelica coordina) |
| 5 | Antes de crear la nueva versión: revisar las **4,568 imágenes sin anotar** que aparecen en Roboflow (45 % del dataset); si entran así, el modelo aprende a ignorar objetos | Juan Diego |
| 6 | Crear la versión nueva con el **mismo augmentation que la v2** (la v3 del 21-10 no tiene augmentation y no es comparable) y reentrenar | Juan Diego |
| 7 | Comparar las métricas nuevas de `spoon`/`fork`/`knife` contra la línea base de la sección 5 | Piero |

