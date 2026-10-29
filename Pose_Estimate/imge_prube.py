import cv2
from matplotlib import image
import mediapipe as mp

# --- Inicialización de MediaPipe Pose ---
mp_pose = mp.solutions.pose #guardar la solución de pose
mp_drawing = mp.solutions.drawing_utils #dibujar la pose para visualizarla

#functions 
def especifics_ponits(body_part, results, width, height):
    if results.pose_landmarks:
        landmark = results.pose_landmarks.landmark[body_part.value]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return x, y
    return None, None

def points_line(first,second, image):
    first_point = cv2.circle(image, first, 5, (255, 0, 0), -1)
    second_point = cv2.circle(image, second, 5, (255, 0, 0), -1)
    line = cv2.line(image, first,second, (255, 255, 255), 2)
    return first_point,second_point, line

#Main

if __name__ == "__main__":
    with mp_pose.Pose(
        static_image_mode = True) as pose:

        image = cv2.imread("image.jpg")
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = pose.process(image_rgb)

        if results.pose_landmarks is not None:
            rigth_knee = especifics_ponits(mp_pose.PoseLandmark.RIGHT_KNEE, results, width, height)
            left_knee = especifics_ponits(mp_pose.PoseLandmark.LEFT_KNEE, results, width, height)

            print(f"Right Knee: {rigth_knee}, Left Knee: {left_knee}")
            points_line(rigth_knee, left_knee, image)
            
            
        cv2.imshow("Detección de Puntos de específicos", image)
        cv2.waitKey(0)

cv2.destroyAllWindows()