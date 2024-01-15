import logging
import os

import customtkinter as ctk
import cv2
import dlib
import numpy as np

import ui
import utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

VERIF_IMG_DIR_PATH = os.path.join("app_data", "verification_images")
INPUT_IMG_DIR_PATH = os.path.join("app_data", "input_image")
INPUT_IMG_PATH = os.path.join(INPUT_IMG_DIR_PATH, "input_image.jpg")
ATTENDANCE_RECORDS_PATH = os.path.join("app_data", "attendance_records.csv")

config = utils.get_config('config.json')
MODEL_PATH = config.get('model_path', '')
EXAMPLE_DATA = config.get('example_data', [])
DETECTION_THRESHOLD = config.get('detection_threshold', 0.0)
VERIFICATION_THRESHOLD = config.get('verification_threshold', 0.0)
VIDEO_SOURCE = config.get('video_source', 0)


class FaceRecognitionAttendance(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.detector = dlib.get_frontal_face_detector()
        self.model = utils.load_model(MODEL_PATH)
        self.capture = utils.setup_video_capture(video_source=VIDEO_SOURCE)
        
        self.web_cam_label = ui.get_web_cam_label(self)
        
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ui.create_window(self, width, height + 180)
        ui.display_verification_button(self, self.verify)
        ui.display_exit_button(self, self.destroy)
        self.status_label = ui.get_status_label(self, text="Начните подтверждение")
        
        self.update()
    
    def verify(self):
        # Преобразование кадра в оттенки серого
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        
        # Обнаружение лиц в кадре
        faces = self.detector(gray)
        
        try:
            face = faces[0]
        except IndexError:
            self.status_label.configure(text="Лицо не обнаружено!")
            return
        
        # Извлечение ограничивающей рамки лица
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        # Проверка, что область лица находится в пределах границ кадра
        if 0 <= x < self.frame.shape[1] and 0 <= y < self.frame.shape[0] and w > 0 and h > 0:
            # Обрезание области лица
            face_roi = self.frame[y:y + h, x:x + w]
            try:
                if face_roi.size != 0:
                    # Изменение размера изображения лица на 250x250px.
                    face_image = cv2.resize(face_roi, (250, 250))
                    cv2.imwrite(INPUT_IMG_PATH, face_image)
            except Exception:
                return
        # Результаты прогнозов
        results = [self.model.predict([np.expand_dims(utils.preprocess(INPUT_IMG_PATH), axis=0),
                                       np.expand_dims(utils.preprocess(os.path.join(VERIF_IMG_DIR_PATH, image)),
                                                      axis=0)]) for image in os.listdir(VERIF_IMG_DIR_PATH)]
        
        # Порог обнаружения: Показатель, прогноз выше которого  считается положительным
        detection = np.sum(np.array(results) > DETECTION_THRESHOLD)
        
        # Порог подтверждения: Доля положительных прогнозов / Общее количество положительных образцов
        verification = detection / len(os.listdir(VERIF_IMG_DIR_PATH))
        verified = verification > VERIFICATION_THRESHOLD
        
        status_label_text = "Подтверждено" if verified else "Не подтверждено"
        self.status_label.configure(text=status_label_text)
        
        logger.info(results)
        logger.info(detection)
        logger.info(f"Точность подтверждения: {verification * 100:.2f}%")
        logger.info(f"Статус подтверждения: {verified}")
        
        if verified:
            utils.register_attendance(*EXAMPLE_DATA, file_path=ATTENDANCE_RECORDS_PATH)
            logger.info(f"Подтврежден {EXAMPLE_DATA[0]}, {EXAMPLE_DATA[1]}, {EXAMPLE_DATA[2]}. ")
    
    def update(self):
        _, self.frame = self.capture.read()
        utils.draw_rectangles_around_faces(self.frame, self.detector)
        frame = utils.convert_to_tkinter_image(self.frame)
        
        # Настройка метки для отображения кадров
        self.web_cam_label.configure(image=frame)
        self.web_cam_label.image = frame
        
        # Обновление кадра каждые 10 мс
        self.after(10, self.update)


if __name__ == "__main__":
    app = FaceRecognitionAttendance()
    app.mainloop()
