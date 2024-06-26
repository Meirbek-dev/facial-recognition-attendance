import logging
import os
import threading
import time

import customtkinter as ctk
import cv2
from deepface.DeepFace import verify, extract_faces

import ui
import utils

ctk.set_appearance_mode("Dark")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

config = utils.load_json("config.json")
VIDEO_SOURCE = config.get("video_source", 0)

INPUT_IMG_PATH = os.path.join("app_data", "input_image", "input_image.jpg")
ATTENDANCE_RECORDS_PATH = os.path.join("app_data", "attendance_records.csv")


class FaceRecognitionAttendance(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.capture = utils.setup_video_capture(video_source=VIDEO_SOURCE)
        
        self.web_cam_label = ui.get_web_cam_label(self)
        
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ui.create_window(self, "Система мониторинга посещаемости", width, height + 240)
        ui.display_verification_button(self, self.verify)
        ui.display_exit_button(self, self.destroy)
        self.id_entry = ui.get_id_entry(self)
        self.status_label = ui.get_status_label(self, text="Начните подтверждение")
        self.progress_bar = ctk.CTkProgressBar(self, mode="indeterminate")
        self.update()
    
    def face_verification_thread(self):
        start_time = time.time()
        try:
            self.status_label.configure(text="Выполняется подтверждение!")
            ui.display_progress_bar(self)
            
            try:
                self.info_label.grid_forget()
            except AttributeError:
                pass
            
            # Обнаружение лиц в кадре
            try:
                faces = extract_faces(self.frame, detector_backend="yolov8")
                face = faces[0]
            except (ValueError, IndexError):
                self.status_label.configure(text="Лицо не обнаружено!")
                logger.error("Лицо не обнаружено!")
                return
            
            cv2.imwrite(INPUT_IMG_PATH, self.frame)
            
            user_id = self.id_entry.get()
            if not user_id.isdigit() and len(user_id) < 6:
                error_msg = "Идентификатор пользователя должен содержать не менее 6 цифр."
                logger.error(error_msg)
                self.status_label.configure(text=error_msg)
                return ValueError(error_msg)
            
            results = verify(INPUT_IMG_PATH, utils.get_verification_image_path(user_id, self.status_label), detector_backend="yolov8",
                             model_name="ArcFace")
            
            logger.info(f"Статус подтверждения: {results['verified']}")
            logger.info(f"Уверенность подтверждения: {face['confidence'] * 100:.2f}%")
            # logger.info(f"Время подтверждения: {results['time']} с")
            
            if not results["verified"]:
                self.status_label.configure(text="Не подтверждено")
                return
            
            user_data = utils.register_attendance(user_id, file_path=ATTENDANCE_RECORDS_PATH)
            self.info_label = ui.get_info_label(self, text=f"{user_data['last_name']} {user_data['first_name']}, "
                                                           f"{user_data['faculty']}, {user_data['group']}", )
            
            self.status_label.configure(text="Подтверждено")
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"Время подтверждения: {elapsed_time:.2f} с")
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
        
        self.after(33, self.update)


if __name__ == "__main__":
    app = FaceRecognitionAttendance()
    app.mainloop()
