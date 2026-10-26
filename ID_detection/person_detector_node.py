"""Person detection/tracking and a heuristic upper-clothing color descriptor."""
import cv2
import numpy as np
from ultralytics import YOLO

COLORS = {
    'roja': (0, 0, 255), 'naranja': (0, 165, 255), 'amarilla': (0, 255, 255),
    'verde': (0, 255, 0), 'azul': (255, 0, 0), 'morada': (255, 0, 255),
    'blanca': (255, 255, 255), 'negra': (0, 0, 0), 'gris': (130, 130, 130),
}

def shirt_color(frame, box):
    """Approximate torso color, using HSV on a central upper-body crop; not clothing segmentation."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    bw, bh = x2-x1, y2-y1
    if bw < 18 or bh < 40:
        return 'desconocida'
    crop = frame[y1+int(.25*bh):y1+int(.53*bh), x1+int(.27*bw):x1+int(.73*bw)]
    if crop.size == 0:
        return 'desconocida'
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    # Quantized histogram filters pixel noise and picks the dominant family.
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    names = np.full(len(H), 'gris', dtype='<U10')
    names[V < 55] = 'negra'
    names[(S < 45) & (V >= 190)] = 'blanca'
    colorful = (S >= 55) & (V >= 55)
    names[colorful & ((H < 9) | (H >= 172))] = 'roja'
    names[colorful & (H >= 9) & (H < 20)] = 'naranja'
    names[colorful & (H >= 20) & (H < 35)] = 'amarilla'
    names[colorful & (H >= 35) & (H < 85)] = 'verde'
    names[colorful & (H >= 85) & (H < 132)] = 'azul'
    names[colorful & (H >= 132) & (H < 172)] = 'morada'
    values, counts = np.unique(names, return_counts=True)
    return str(values[np.argmax(counts)])

class PersonDetectorNode:
    def __init__(self, model='yolo11n.pt', confidence=.4):
        self.model = YOLO(model)  # downloads weights on first run
        self.confidence = confidence

    def process(self, frame):
        result = self.model.track(frame, persist=True, classes=[0], conf=self.confidence,
                                  tracker='bytetrack.yaml', verbose=False)[0]
        people = []
        for box in result.boxes:
            if box.id is None:
                continue
            coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
            track_id = int(box.id[0].item())
            x1, y1, x2, y2 = coords
            people.append({'track_id': track_id, 'box': coords,
                           'center_px': [(x1+x2)//2, (y1+y2)//2],
                           'confidence': round(float(box.conf[0].item()), 3),
                           'camiseta': shirt_color(frame, coords)})
        return people
