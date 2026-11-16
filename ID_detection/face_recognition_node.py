
"""Reconocimiento facial RoboCup con OpenCV YuNet + SFace."""

import json
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np


MODELS = {
    "face_detection_yunet_2023mar.onnx":
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",

    "face_recognition_sface_2021dec.onnx":
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

# CARGAR MODELOS
def get_model(name, folder):

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    dest = folder / name

    min_size = (
        10_000_000 if "sface" in name else 100_000
    )

    if dest.exists() and dest.stat().st_size >= min_size:
        return str(dest)

    url = (
        "https://raw.githubusercontent.com/"
        "opencv/opencv_zoo/main/models/"
        + MODELS[name]
    )

    print(f"Descargando modelo {name} ...")

    req = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    try:
        with urlopen(req, timeout=120) as response:
            with open(dest, "wb") as output:

                while True:
                    block = response.read(1024 * 1024)

                    if not block:
                        break

                    output.write(block)

        if dest.stat().st_size < min_size:
            raise RuntimeError(
                "Archivo descargado incompleto"
            )

    except Exception as error:

        dest.unlink(missing_ok=True)

        raise RuntimeError(
            f"No se pudo descargar {name}: {error}"
        ) from error

    return str(dest)

# CLASE DE RECONOCIMIENTO FACIAL
class FaceRecognitionNode:

    def __init__(
        self,
        data_dir="face_data",
        threshold=0.45
    ):

        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)

        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        det = get_model(
            "face_detection_yunet_2023mar.onnx",
            model_dir
        )

        rec = get_model(
            "face_recognition_sface_2021dec.onnx",
            model_dir
        )

        # Detector facial YuNet
        self.detector = cv2.FaceDetectorYN.create(
            det,
            "",
            (320, 320),
            0.80,
            0.3,
            5000
        )

        # Reconocedor facial SFace
        self.recognizer = cv2.FaceRecognizerSF.create(
            rec,
            ""
        )

        self.threshold = threshold

        # IDs faciales y vectores registrados
        self.known = {}

        # Archivo de nombres recibidos desde STT
        self.names_path = self.root / "nombres.json"

        if self.names_path.exists():

            try:
                self.names = json.loads(
                    self.names_path.read_text(
                        encoding="utf-8"
                    )
                )

                if not isinstance(self.names, dict):
                    raise ValueError(
                        "El archivo de nombres no es un diccionario"
                    )

            except (OSError, ValueError) as error:
                raise RuntimeError(
                    f"Error al cargar nombres.json: {error}"
                ) from error

        else:
            self.names = {}

        self.train()

        print("[FACE] Reconocimiento facial inicializado")
        print("[FACE] IDs cargados:", list(self.known.keys()))
        print("[FACE] Nombres:", self.names)

    # VALIDAR VECTOR FACIAL
    def validate_embedding(self, face):

        if face is None:
            return None

        # Evitar guardar objetos OpenCV como archivos NumPy
        if not isinstance(face, np.ndarray):

            print(
                "[FACE ERROR] Se esperaba un arreglo NumPy, "
                f"pero se recibió: {type(face)}"
            )

            return None

        try:
            vector = np.asarray(
                face,
                dtype=np.float32
            ).reshape(-1)

        except (TypeError, ValueError):
            print("[FACE ERROR] Vector facial inválido")
            return None

        # SFace genera vectores de 128 componentes
        if vector.size != 128:

            print(
                "[FACE ERROR] Dimensión incorrecta:",
                vector.shape
            )

            return None

        if not np.all(np.isfinite(vector)):
            print("[FACE ERROR] Vector con valores inválidos")
            return None

        norm = np.linalg.norm(vector)

        if not np.isfinite(norm) or norm <= 0:
            print("[FACE ERROR] Norma inválida")
            return None

        return (vector / norm).astype(np.float32)

    # EXTRAER CARACTERÍSTICAS FACIALES
    def extract(self, frame, box):

        if frame is None or not isinstance(frame, np.ndarray):
            return None

        x1, y1, x2, y2 = [int(v) for v in box]

        h, w = frame.shape[:2]

        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))

        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        # Buscar el rostro en la parte superior del cuerpo
        upper_height = max(
            1,
            int((y2 - y1) * 0.6)
        )

        roi = frame[
            y1:min(h, y1 + upper_height),
            x1:x2
        ]

        rh, rw = roi.shape[:2]

        if rh < 45 or rw < 45:
            return None

        # Detectar rostro con YuNet
        self.detector.setInputSize((rw, rh))

        _, detected_faces = self.detector.detect(roi)

        if detected_faces is None:
            return None

        if len(detected_faces) == 0:
            return None

        # Seleccionar el rostro de mayor área
        detected_face = max(
            detected_faces,
            key=lambda f: float(f[2] * f[3])
        )

        try:

            # Alinear rostro
            aligned = self.recognizer.alignCrop(
                roi,
                detected_face
            )

            # Extraer embedding SFace
            embedding = self.recognizer.feature(
                aligned
            )

            # Validar y normalizar
            return self.validate_embedding(embedding)

        except cv2.error as error:

            print(
                f"[FACE ERROR] Extracción facial: {error}"
            )

            return None

    # CARGAR ROSTROS REGISTRADOS
    def train(self):

        self.known = {}

        for folder in self.root.glob("id_*"):

            if not folder.is_dir():
                continue

            if not folder.name[3:].isdigit():
                continue

            person_id = int(folder.name[3:])

            embeddings = []

            for file in sorted(folder.glob("*.npy")):

                try:
                    vector = np.load(
                        file,
                        allow_pickle=False
                    )

                    vector = self.validate_embedding(
                        vector
                    )

                    if vector is not None:
                        embeddings.append(vector)

                except (ValueError, OSError) as error:

                    print(
                        f"[FACE] Muestra ignorada {file.name}: "
                        f"{error}"
                    )

            if embeddings:
                self.known[person_id] = embeddings

    # IDENTIFICAR UNA PERSONA
    def identify(self, face):

        face = self.validate_embedding(face)

        if face is None:
            return None

        if not self.known:
            return None

        best_id = None
        best_score = -1.0

        for person_id, samples in self.known.items():

            score = max(
                float(np.dot(face, sample))
                for sample in samples
            )

            if score > best_score:

                best_id = person_id
                best_score = score

        if best_score >= self.threshold:
            return best_id

        return None

    # REGISTRAR ROSTRO
    def enroll(self, face, person_id):

        # Comprobar que se recibió un embedding válido
        vector = self.validate_embedding(face)

        if vector is None:

            print(
                "[FACE ERROR] No se pudo registrar "
                "el rostro: embedding inválido."
            )

            return False

        person_id = int(person_id)

        directory = self.root / f"id_{person_id}"

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        # Buscar el siguiente nombre de archivo disponible
        count = 0

        while (directory / f"{count:04d}.npy").exists():
            count += 1

        filepath = directory / f"{count:04d}.npy"

        try:

            np.save(
                filepath,
                vector,
                allow_pickle=False
            )

        except (OSError, ValueError, TypeError) as error:

            print(
                f"[FACE ERROR] No se pudo guardar: {error}"
            )

            return False

        # Actualizar memoria facial
        self.train()

        print(
            f"[FACE] Muestra guardada para ID {person_id}"
        )

        return True

    # ASIGNAR NOMBRE RECIBIDO DESDE STT
    def assign_name(self, person_id, nombre):

        nombre = str(nombre).strip()

        if not nombre:
            print("[FACE] Nombre vacío")
            return False

        person_id = int(person_id)

        if person_id not in self.known:

            print(
                f"[FACE] El ID {person_id} "
                "no tiene un rostro registrado."
            )

            return False

        # Conservar el estado anterior ante errores de guardado
        updated_names = dict(self.names)
        updated_names[str(person_id)] = nombre

        try:

            temporary = self.names_path.with_suffix(".tmp")

            temporary.write_text(
                json.dumps(
                    updated_names,
                    ensure_ascii=False,
                    indent=4
                ),
                encoding="utf-8"
            )

            temporary.replace(self.names_path)

        except OSError as error:

            print(
                f"[FACE ERROR] No se pudo guardar nombre: {error}"
            )

            return False

        self.names = updated_names

        print(
            f"[FACE] ID {person_id} asociado a {nombre}"
        )

        return True

    # RECUPERAR NOMBRE POR ID
    def get_name(self, person_id):

        if person_id is None:
            return None

        return self.names.get(str(person_id))