import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import cv2
from person_detector_node import PersonDetectorNode
from face_recognition_node import FaceRecognitionNode

# Archivo compartido con el STT de Sebastian
STT_FILE = Path("nombre_stt.json")

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--threshold", type=float, default=0.45)
    args = ap.parse_args()

    # Inicializar deteccion y reconocimiento facial
    detector = PersonDetectorNode(args.model)
    faces = FaceRecognitionNode(threshold=args.threshold)

    # Verificar nombres registrados
    print("=== ROBOCUP VISION ===")
    print("Nombres cargados:", faces.names)
    print("Nombre del ID 1:", faces.get_name(1))

    # Inicializar camara en Windows
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir la camara. Revisa --camera y los permisos."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Variables del sistema
    track_identity = {}
    colors = defaultdict(lambda: defaultdict(Counter))
    selected_index = 0
    nombre_pendiente = None

    print("TAB: seleccionar persona")
    print("E: registrar rostro y asignar nombre")
    print("Q: salir")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("No se pudo leer un fotograma de la camara.")
                break

            # RECIBIR NOMBRE DESDE EL STT
            if STT_FILE.exists():
                try:
                    data = json.loads(
                        STT_FILE.read_text(encoding="utf-8")
                    )

                    nombre_recibido = str(
                        data.get("nombre", "")
                    ).strip()

                    if nombre_recibido:
                        nombre_pendiente = nombre_recibido

                        print(
                            f"[STT] Nombre recibido: {nombre_pendiente}"
                        )

                    STT_FILE.unlink()

                except (json.JSONDecodeError, OSError) as error:
                    print(f"[STT] Error: {error}")

            # DETECTAR PERSONAS
            persons = detector.process(frame)
            persons.sort(key=lambda person: person["track_id"])
            selected_index = min(
                selected_index,
                max(0, len(persons) - 1)
            )
            sample = None

            # RECONOCIMIENTO FACIAL
            for idx, p in enumerate(persons):
                track = p["track_id"]

                face = faces.extract(frame, p["box"])
                recognized = faces.identify(face)

                if recognized is not None:
                    track_identity[track] = recognized

                elif face is not None:
                    track_identity.pop(track, None)

                identity = track_identity.get(track)

                # Guardar caracteristica visual
                if identity is not None:
                    colors[identity]["camiseta"][p["camiseta"]] += 1

                x1, y1, x2, y2 = p["box"]
                active = idx == selected_index

                color_box = (
                    (0, 255, 255)
                    if active
                    else (0, 220, 0)
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color_box,
                    2
                )

                # MOSTRAR NOMBRE EN LA CAMARA
                if identity is not None:
                    nombre = faces.get_name(identity)

                    if nombre:
                        pid = f"{nombre} (ID {identity})"
                    else:
                        pid = f"ID {identity}"
                else:
                    pid = "Desconocida"

                label = (
                    f'Track {track} | {pid} | '
                    f'camiseta {p["camiseta"]}'
                )

                cv2.putText(
                    frame,
                    label,
                    (max(0, x1), max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

                # Guardar rostro de persona seleccionada
                if active:
                    sample = face

            #  MOSTRAR INSTRUCCIONES
            cv2.putText(
                frame,
                "TAB: elegir | E: registrar | Q: salir",
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2
            )

            cv2.imshow("RoboCup Vision - prototipo", frame)

            key = cv2.waitKey(1) & 0xFF

            # CONTROLES DEL USUARIO
            if key in (ord("q"), 27):
                break

            # Cambiar persona seleccionada
            if key == 9 and persons:
                selected_index = (
                    selected_index + 1
                ) % len(persons)

            # Registrar rostro
            if key == ord("e"):

                if not persons or sample is None:
                    print(
                        "No hay un rostro frontal detectable "
                        "en la persona seleccionada."
                    )
                    continue

                track = persons[selected_index]["track_id"]

                existing = track_identity.get(track)

                # Crear ID cuando la persona es nueva
                if existing is None:
                    existing = max(
                        [
                            int(d.name[3:])
                            for d in faces.root.glob("id_*")
                            if d.is_dir()
                            and d.name[3:].isdigit()
                        ] or [0]
                    ) + 1

                # Registrar muestra facial
                if faces.enroll(sample, existing):

                    track_identity[track] = existing

                    print(
                        f"[ROBOCUP] Rostro registrado con ID {existing}"
                    )

                    # Asignar nombre recibido del STT
                    if nombre_pendiente is not None:

                        if faces.assign_name(
                            existing,
                            nombre_pendiente
                        ):

                            print(
                                f"[ROBOCUP] ID {existing} "
                                f"asociado a {nombre_pendiente}"
                            )

                            nombre_pendiente = None

                    else:
                        nombre_actual = faces.get_name(existing)

                        if nombre_actual:
                            print(
                                f"[ROBOCUP] Persona reconocida: "
                                f"{nombre_actual}"
                            )
                        else:
                            print(
                                "[ROBOCUP] Sin nombre recibido del STT."
                            )

            # GUARDAR DETECCIONES EN JSON
            snapshot = []

            for p in persons:
                identity = track_identity.get(p["track_id"])

                snapshot.append({
                    **p,
                    "person_id": identity,
                    "nombre": faces.get_name(identity)
                })

            Path("detecciones.json").write_text(
                json.dumps(
                    snapshot,
                    ensure_ascii=False,
                    indent=2
                ),
                encoding="utf-8"
            )

    finally:
        # CERRAR PROGRAMA Y GUARDAR MEMORIA
        cap.release()
        cv2.destroyAllWindows()

        summary = {
            str(pid): {
                feature: votes.most_common(1)[0][0]
                for feature, votes in v.items()
                if votes
            }
            for pid, v in colors.items()
        }

        Path("personas_memoria.json").write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        print("Resumen guardado en personas_memoria.json")


if __name__ == "__main__":
    run()
