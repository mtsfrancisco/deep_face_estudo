import cv2
import os
import time
from deepface import DeepFace
import face_recognition
import json

# Caminho para a pasta "users"
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
USERS_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "..", "local_database", "users")

class Person:
    def __init__(self, name, encoding, image):
        self.name = name
        self.encoding = encoding
        self.image = image


class cam_face_recognition:
    def __init__(self, wait_time=5):
        self.wait_time = wait_time
        self.last_check_time = time.time() - wait_time
        self.video_capture = cv2.VideoCapture(0)
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.square_size = 500
        self.x_start = self.frame_width // 2 - self.square_size // 2
        self.y_start = self.frame_height // 2 - self.square_size // 2
        self.x_end = self.x_start + self.square_size
        self.y_end = self.y_start + self.square_size

    def draw_square(self, frame):
        """Desenha um quadrado no meio da tela."""
        cv2.rectangle(frame, (self.x_start, self.y_start), (self.x_end, self.y_end), (0, 255, 0), 2)

    def analyze_face(self, roi):
        """Analisa a face na região de interesse (ROI) usando DeepFace."""
        temp_roi_path = "temp_roi.jpg"
        cv2.imwrite(temp_roi_path, roi)
        try:
            analysis = DeepFace.analyze(img_path=temp_roi_path, actions=["age", "gender", "race", "emotion"], enforce_detection=False)
            if analysis:
                return analysis[0]
        except Exception as e:
            print(f"Erro ao analisar a face: {e}")
        finally:
            if os.path.exists(temp_roi_path):
                os.remove(temp_roi_path)
        return None

    def recognize_face(self, frame):
        """Reconhece a face na região de interesse (ROI) usando face_recognition."""
        dfs = DeepFace.find(
        model_name='Facenet512',
        img_path = frame, 
        db_path = "local_database/users", 
        detector_backend = 'ssd',
        align = True,
    )
        if dfs:
            return dfs[0].identity
        return None

    def display_person_info(self, frame, person):
        """Exibe informações da pessoa reconhecida no frame."""
        person_name = person[0].split("/")[-1].split(".")[0]
        person_image = cv2.resize(cv2.imread(str(person[0])), (self.square_size, self.square_size))
        frame[0:self.square_size, 0:self.square_size] = person_image
        frame[0:self.square_size, self.square_size:self.square_size * 2] = cv2.resize(frame[self.y_start:self.y_end, self.x_start:self.x_end], (self.square_size, self.square_size))
        cv2.putText(frame, person_name, (10, self.square_size + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        print(f"Pessoa reconhecida: {person_name}")

    def display_unknown_person_info(self, frame, analysis):
        """Exibe informações de uma pessoa não reconhecida no frame."""
        cv2.putText(frame, "Pessoa nao conhecida", (10, self.square_size + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Age: {analysis['age']}", (10, self.square_size + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Gender: {analysis['dominant_gender']}", (10, self.square_size + 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Race: {analysis['dominant_race']}", (10, self.square_size + 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Emotion: {analysis['dominant_emotion']}", (10, self.square_size + 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        print("Pessoa não reconhecida")

    def run(self):
        """Executa o loop principal de detecção e reconhecimento de faces."""
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            self.draw_square(frame)

            if time.time() - self.last_check_time >= self.wait_time:
                roi = frame[self.y_start:self.y_end, self.x_start:self.x_end]
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_roi)
                if face_encodings:
                    result_recognition_result = self.recognize_face(roi)
                    #print(result_recognition_result[0])
                    self.display_person_info(frame, result_recognition_result)
                    cv2.imshow("Webcam", frame)
                    cv2.waitKey(4000)
                else:
                    print("Nenhuma face detectada.")

                self.last_check_time = time.time()

            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()


def main():
    face_recognizer = cam_face_recognition()
    face_recognizer.run()

if __name__ == "__main__":
    main()




