"""Reconocimiento facial con OpenCV YuNet + SFace (registro local y voluntario)."""
from pathlib import Path
from urllib.request import Request, urlopen
import cv2
import numpy as np

MODELS = {
    'face_detection_yunet_2023mar.onnx': 'face_detection_yunet/face_detection_yunet_2023mar.onnx',
    'face_recognition_sface_2021dec.onnx': 'face_recognition_sface/face_recognition_sface_2021dec.onnx',
}


def get_model(name, folder):
    dest = folder / name
    if dest.exists() and dest.stat().st_size > 100_000:
        return str(dest)
    # GitHub redirect resolves files stored with Git LFS; avoids shipping model binaries.
    url = 'https://github.com/opencv/opencv_zoo/raw/main/models/' + MODELS[name]
    print(f'Descargando modelo {name} ...')
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urlopen(req, timeout=120) as response, open(dest, 'wb') as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
        min_size = 10_000_000 if 'sface' in name else 100_000
        if dest.stat().st_size < min_size:
            raise RuntimeError('Archivo descargado incompleto')
    except Exception:
        dest.unlink(missing_ok=True)
        raise RuntimeError('No se pudo descargar el modelo. Descargalo desde OpenCV Zoo y guardalo en models/.')
    return str(dest)


class FaceRecognitionNode:
    def __init__(self, data_dir='face_data', threshold=0.45):
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        model_dir = Path('models')
        model_dir.mkdir(exist_ok=True)
        det = get_model('face_detection_yunet_2023mar.onnx', model_dir)
        rec = get_model('face_recognition_sface_2021dec.onnx', model_dir)
        self.detector = cv2.FaceDetectorYN.create(det, '', (320, 320), 0.80, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(rec, '')
        self.threshold = threshold  # cosine similarity: mayor = mas parecido
        self.known = {}
        self.train()

    def extract(self, frame, box):
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        # Buscar en el 60 % superior de la persona.
        roi = frame[y1:y1 + max(1, int((y2-y1)*0.6)), x1:x2]
        rh, rw = roi.shape[:2]
        if rh < 45 or rw < 45:
            return None
        self.detector.setInputSize((rw, rh))
        _, faces = self.detector.detect(roi)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda f: float(f[2] * f[3]))
        aligned = self.recognizer.alignCrop(roi, face)
        embedding = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else None

    def train(self):
        self.known = {}
        for folder in self.root.glob('id_*'):
            if not folder.is_dir() or not folder.name[3:].isdigit():
                continue
            embeddings = []
            for file in folder.glob('*.npy'):
                try:
                    vector = np.load(file, allow_pickle=False).reshape(-1).astype(np.float32)
                    if vector.size == 128 and np.linalg.norm(vector) > 0:
                        embeddings.append(vector / np.linalg.norm(vector))
                except (ValueError, OSError):
                    continue
            if embeddings:
                self.known[int(folder.name[3:])] = embeddings

    def identify(self, face):
        if face is None or not self.known:
            return None
        best_id, best_score = None, -1.0
        for person_id, samples in self.known.items():
            score = max(float(np.dot(face, sample)) for sample in samples)
            if score > best_score:
                best_id, best_score = person_id, score
        return best_id if best_score >= self.threshold else None

    def enroll(self, face, person_id):
        if face is None:
            return False
        directory = self.root / f'id_{person_id}'
        directory.mkdir(parents=True, exist_ok=True)
        count = len(list(directory.glob('*.npy')))
        np.save(directory / f'{count:04d}.npy', face)
        self.train()
        return True
