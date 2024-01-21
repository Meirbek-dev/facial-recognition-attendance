import logging
import os
import threading

import customtkinter as ctk
import cv2
from deepface.DeepFace import verify, extract_faces

import ui
import utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

config = utils.load_json('config.json')
EXAMPLE_ID = config.get("example_id", "")
VIDEO_SOURCE = config.get('video_source', 0)

VERIF_IMGS_DIR_PATH = os.path.join("app_data", "verification_data")
VERIF_IMG_PATH = os.path.join(VERIF_IMGS_DIR_PATH, EXAMPLE_ID, "verification_image.jpg")
INPUT_IMG_PATH = os.path.join("app_data", "input_image", "input_image.jpg")
ATTENDANCE_RECORDS_PATH = os.path.join("app_data", "attendance_records.csv")


class FaceRecognitionAttendance(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.capture = utils.setup_video_capture(video_source=VIDEO_SOURCE)
        
        self.web_cam_label = ui.get_web_cam_label(self)
        
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ui.create_window(self, width, height + 180)
        ui.display_verification_button(self, self.verify)
        ui.display_exit_button(self, self.destroy)
        self.status_label = ui.get_status_label(self, text="Начните подтверждение")
        self.progress_bar = ctk.CTkProgressBar(self, mode='indeterminate')
        self.update()
    
    def face_verification_thread(self):
        try:
            self.status_label.configure(text="Выполняется подтверждение!")
            ui.display_progress_bar(self)
            
            try:
                self.info_label.grid_forget()
            except AttributeError:
                pass
            
            # Обнаружение лиц в кадре
            try:
                faces = extract_faces(self.frame, detector_backend="mtcnn")
                face = faces[0]
            except ValueError as e:
                self.status_label.configure(text="Лицо не обнаружено!")
                logger.error(e)
                return
            
            cv2.imwrite(INPUT_IMG_PATH, self.frame)
            
            results = verify(INPUT_IMG_PATH, VERIF_IMG_PATH, detector_backend="mtcnn", model_name="Facenet512")
            
            logger.info(f"Статус подтверждения: {results['verified']}")
            logger.info(f"Уверенность подтверждения: {face['confidence'] * 100:.2f}%")
            
            if not results['verified']:
                self.status_label.configure(text="Не подтверждено")
                return
            
            user_data = utils.register_attendance(EXAMPLE_ID, file_path=ATTENDANCE_RECORDS_PATH)
            self.info_label = ui.get_info_label(self, text=f"{user_data['last_name']} {user_data['first_name']}, "
                                                           f"{user_data['faculty']}, {user_data['group']}")
            
            self.status_label.configure(text="Подтверждено")
        except Exception as e:
            logger.error(f"Ошибка во время подтверждения: {e}")
        finally:
            ui.hide_progress_bar(self)
    
    def verify(self):
        verification_thread = threading.Thread(target=self.face_verification_thread, args=())
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
