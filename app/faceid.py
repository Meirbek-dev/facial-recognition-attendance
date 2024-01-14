from kivy.config import Config

Config.set('graphics', 'resizable', False)
Config.set('graphics', 'maximizable', False)

import os

import cv2
import dlib
import numpy as np
import tensorflow as tf
from PIL import Image as PILImage
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout as BoxLayout
from kivymd.uix.button import MDRaisedButton as Button
from kivymd.uix.label import MDLabel as Label

from layers import L1Dist
from utils import register_attendance, preprocess, get_config

INPUT_IMG_DIR_PATH = os.path.join("app_data", "input_image")
VERIF_IMG_DIR_PATH = os.path.join("app_data", "verification_images")
INPUT_IMG_PATH = os.path.join(INPUT_IMG_DIR_PATH, "input_image.jpg")
ATTENDANCE_RECORDS_PATH = os.path.join("app_data", "attendance_records.csv")

config = get_config('config.json')

MODEL_PATH = config.get('model_path', '')
EXAMPLE_DATA = config.get('example_data', [])

DETECTION_THRESHOLD = config.get('detection_threshold', 0.0)
VERIFICATION_THRESHOLD = config.get('verification_threshold', 0.0)


class FaceIDApp(MDApp):
    def build(self):
        self.title = "Система распознавания лиц"
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Gray"
        
        # Главные компоненты
        self.web_cam = Image(size_hint=(1, 0.8))
        self.button = Button(text="Подтвердить", on_press=self.verify, size_hint=(1, 0.1), font_size="16sp",
                             font_style="Body1", md_bg_color=(255, 255, 255, 1))
        self.verification_label = Label(text="Начните подтверждение", size_hint=(1, 0.1), halign="center")
        
        # Размещение элементов на макете
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)
        
        # Загрузка модели
        self.model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"L1Dist": L1Dist})
        
        # Загрузка предварительно обученной модель распознавания лицы
        self.detector = dlib.get_frontal_face_detector()
        
        # Настройка устройства видео-захвата
        self.capture = cv2.VideoCapture(0)
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Window.size = (width, height + 120)
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        
        return layout
    
    # Непрерывное считывание изображения с веб-камеры.
    def update(self, *args):
        _, frame = self.capture.read()
        
        # Преобразование кадра в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Обнаружение лиц в кадре
        self.faces = self.detector(gray)
        
        for face in self.faces:
            # Извлечение ограничивающей рамки лица
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            
            # Проверка, что область лица находится в пределах границ кадра
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0] and w > 0 and h > 0:
                # Обрезание области лица
                self.face_roi = frame[y:y + h, x:x + w]
                
                if self.face_roi.size != 0:
                    # Обрисовка прямоугольника вокруг обнаруженного лица
                    cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
        
        # Переворачивание по горизонтали и преобразовние изображения в текстуру
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        img_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.web_cam.texture = img_texture
    
    def verify(self, model):
        try:
            if self.face_roi.size != 0:
                # Изменение размера изображения лица на 250x250px.
                face_image = cv2.resize(self.face_roi, (250, 250))
                self.face_roi = None
                # Преобразование изображения из BGR в RGB (OpenCV использует BGR, а Pillow использует RGB).
                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                
                # Создание Pillow-изображения из массива NumPy.
                pil_image = PILImage.fromarray(face_image_rgb)
                
                pil_image.save(str(INPUT_IMG_PATH))
        except AttributeError:
            self.verification_label.text = "Лицо не обнаружено!"
            return
        # Построение массива результатов прогнозов
        results = []
        for image in os.listdir(VERIF_IMG_DIR_PATH):
            input_img = preprocess(INPUT_IMG_PATH)
            validation_img = preprocess(os.path.join(VERIF_IMG_DIR_PATH, image))
            
            # Результаты прогнозов
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        # Порог обнаружения: Показатель, прогноз выше которого  считается положительным
        detection = np.sum(np.array(results) > DETECTION_THRESHOLD)
        
        # Порог подтверждения: Доля положительных прогнозов / Общее количество положительных образцов
        verification = detection / len(os.listdir(VERIF_IMG_DIR_PATH))
        verified = verification > VERIFICATION_THRESHOLD
        
        if verified:
            register_attendance(*EXAMPLE_DATA, file_path=ATTENDANCE_RECORDS_PATH)
            Logger.info(f"Подтврежден {EXAMPLE_DATA[0]}, {EXAMPLE_DATA[1]}, {EXAMPLE_DATA[2]}. ")
            verification_label_text = "Подтверждено"
        else:
            verification_label_text = "Не подтверждено"
        
        self.verification_label.text = verification_label_text
        
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)
        
        return results, verified


if __name__ == "__main__":
    FaceIDApp().run()
