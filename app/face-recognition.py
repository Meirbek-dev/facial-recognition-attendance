import logging
import os
import threading

import customtkinter as ctk
import cv2
import dlib
from deepface import DeepFace

import ui
import utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

VERIF_IMG_DIR_PATH = os.path.join("app_data", "verification_images")
INPUT_IMG_PATH = os.path.join("app_data", "input_image", "input_image.jpg")
ATTENDANCE_RECORDS_PATH = os.path.join("app_data", "attendance_records.csv")

config = utils.get_config('config.json')
EXAMPLE_DATA = config.get('example_data', [])
DETECTION_THRESHOLD = config.get('detection_threshold', 0.0)
VERIFICATION_THRESHOLD = config.get('verification_threshold', 0.0)
VIDEO_SOURCE = config.get('video_source', 0)


class FaceRecognitionAttendance(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.detector = dlib.get_frontal_face_detector()
        self.capture = utils.setup_video_capture(video_source=VIDEO_SOURCE)
        
        self.web_cam_label = ui.get_web_cam_label(self)
        
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ui.create_window(self, width, height + 140)
        ui.display_verification_button(self, self.verify)
        ui.display_exit_button(self, self.destroy)
        self.status_label = ui.get_status_label(self, text="Начните подтверждение")
        
        self.update()
    
    def face_verification_thread(self, face):
        try:
            # Results of DeepFace.verify
            results = [
                DeepFace.verify(INPUT_IMG_PATH, os.path.join(VERIF_IMG_DIR_PATH, image), detector_backend="mtcnn",
                                model_name="Facenet512") for image in os.listdir(VERIF_IMG_DIR_PATH)]
            
            logger.info(f"Статус подтверждения: {results[0]['verified']}")
            logger.info(f"Уверенность подтверждения: {face['confidence'] * 10:.2f}%")
            
            if not results[0]['verified']:
                self.status_label.configure(text="Не подтверждено")
                return
            
            utils.register_attendance(*EXAMPLE_DATA, file_path=ATTENDANCE_RECORDS_PATH)
            logger.info(f"Подтврежден {EXAMPLE_DATA[0]}, {EXAMPLE_DATA[1]}, {EXAMPLE_DATA[2]}. ")
            self.status_label.configure(text="Подтверждено")
        except Exception as e:
            logger.error(f"Ошибка во время подтверждения: {e}")
    
    def verify(self):
        # Обнаружение лиц в кадре
        try:
            faces = DeepFace.extract_faces(self.frame)
            face = faces[0]
        except ValueError as e:
            self.status_label.configure(text="Лицо не обнаружено!")
            logger.error(e)
            return
        
        cv2.imwrite(INPUT_IMG_PATH, self.frame)
        
        verification_thread = threading.Thread(target=self.face_verification_thread, args=(face,))
        verification_thread.start()
    
    def update(self):
        _, self.frame = self.capture.read()
        utils.draw_rectangle_around_face(self.frame)
        frame = utils.convert_to_tkinter_image(self.frame)
        
        # Настройка метки для отображения кадров
        self.web_cam_label.configure(image=frame)
        self.web_cam_label.image = frame
        
        # Обновление кадра каждые 10 мс
        self.after(10, self.update)


if __name__ == "__main__":
    app = FaceRecognitionAttendance()
    app.mainloop()
