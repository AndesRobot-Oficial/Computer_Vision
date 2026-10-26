"""Webcam demo. E=collect face sample of selected track, TAB=next track, Q=quit."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import cv2
from person_detector_node import PersonDetectorNode
from face_recognition_node import FaceRecognitionNode


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--model', default='yolo11n.pt')
    ap.add_argument('--threshold', type=float, default=0.45)
    args = ap.parse_args()
    detector = PersonDetectorNode(args.model)
    faces = FaceRecognitionNode(threshold=args.threshold)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError('No se pudo abrir la camara. Revisa --camera y los permisos.')
    # Keep identity stable across small face-detection gaps while current track is active.
    track_identity = {}
    colors = defaultdict(lambda: defaultdict(Counter))
    selected_index = 0
    print('TAB: seleccionar persona | E: guardar muestra de rostro (con permiso) | Q: salir')
    print('Muestra el rostro varias veces al pulsar E para mejorar el registro.')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            persons = detector.process(frame)
            persons.sort(key=lambda person: person['track_id'])
            selected_index = min(selected_index, max(0,len(persons)-1))
            sample = None
            for idx, p in enumerate(persons):
                track = p['track_id']
                face = faces.extract(frame, p['box'])
                recognized = faces.identify(face)
                if recognized is not None:
                    track_identity[track] = recognized
                elif face is not None:
                    track_identity.pop(track, None)  # rostro visible, pero no identificado
                identity = track_identity.get(track)
                if identity is not None:
                    colors[identity]['camiseta'][p['camiseta']] += 1
                x1,y1,x2,y2 = p['box']
                active = idx == selected_index
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255) if active else (0,220,0),2)
                pid = f'ID {identity}' if identity is not None else 'desconocida'
                label = f'Track {track} | {pid} | camiseta {p["camiseta"]}'
                cv2.putText(frame,label,(max(0,x1),max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
                if active:
                    sample = face
            cv2.putText(frame,'TAB: elegir | E: registrar rostro | Q: salir',
                        (12,25),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),2)
            cv2.imshow('RoboCup Vision - prototipo',frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'),27):
                break
            if key == 9 and persons:
                selected_index = (selected_index+1)%len(persons)
            if key == ord('e'):
                if not persons or sample is None:
                    print('No hay un rostro frontal detectable en la persona seleccionada.')
                    continue
                track = persons[selected_index]['track_id']
                existing = track_identity.get(track)
                if existing is None:
                    existing = max([int(d.name[3:]) for d in faces.root.glob('id_*')
                                    if d.is_dir() and d.name[3:].isdigit()] or [0]) + 1
                if faces.enroll(sample, existing):
                    track_identity[track] = existing
                    print(f'Muestra registrada para ID {existing}. Pulsa E varias veces.')
            # Snapshot for future integration; tracking coordinates are image pixels, not 3D positions.
            snapshot = []
            for p in persons:
                identity = track_identity.get(p['track_id'])
                snapshot.append({**p,'person_id':identity})
            Path('detecciones.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
    finally:
        cap.release()
        cv2.destroyAllWindows()
        summary = {str(pid): {feature: votes.most_common(1)[0][0] for feature,votes in v.items() if votes}
                   for pid,v in colors.items()}
        Path('personas_memoria.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        print('Resumen guardado en personas_memoria.json')

if __name__ == '__main__':
    run()
