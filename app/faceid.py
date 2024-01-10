import os

import cv2
import numpy as np
import tensorflow as tf
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label

from layers import L1Dist

MODEL_PATH = "siamese_model.keras"

INPUT_IMG_DIR_PATH = os.path.join("application_data", "input_image")
VERIF_IMG_DIR_PATH = os.path.join("application_data", "verification_images")
INPUT_IMG_PATH = os.path.join(INPUT_IMG_DIR_PATH, "input_image.jpg")

Window.maximize()


# Загрузка изображения из файла и конвертация в 105x105px
def preprocess(file_path):
    # Чтение изображения
    byte_img = tf.io.read_file(file_path)
    # Загрузка изображения
    img = tf.io.decode_jpeg(byte_img)
    # Изменение размера изображения на 105x105
    img = tf.image.resize(img, (105, 105))
    # Масштабирование изображения в диапазоне от 0 до 1
    img /= 255.0
    return img


class CamApp(App):
    def build(self):
        self.title = "Система распознавания лиц"

        # Главные компоненты
        self.web_cam = Image(size_hint=(1, 0.8))
        self.button = Button(text="Подтвердить", on_press=self.verify, size_hint=(1, 0.1))
        self.verification_label = Label(text="Начните подтверждение", size_hint=(1, 0.1))

        # Размещение элементов на макете
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Загрузка модели
        self.model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"L1Dist": L1Dist})

        # Настройка устройства видео-захвата
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout

    # Непрерывное считывание изображения с веб-камеры.
    def update(self, *args):
        _, frame = self.capture.read()
        frame = frame[115: 115 + 250, 195: 195 + 250, :]

        # Переворачивание по горизонтали и преобразовние изображения в текстуру
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        img_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.web_cam.texture = img_texture

    def verify(self, model, detection_threshold=0.8, verification_threshold=0.6):
        # Построение массива результатов прогнозов
        results = []
        for image in os.listdir(VERIF_IMG_DIR_PATH):
            input_img = preprocess(INPUT_IMG_PATH)
            validation_img = preprocess(os.path.join(VERIF_IMG_DIR_PATH, image))

            # Результаты прогнозов
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        # Порог обнаружения: Показатель, прогноз выше которого  считается положительным
        detection = np.sum(np.array(results) > detection_threshold)

        # Порог подтверждения: Доля положительных прогнозов / Общее количество положительных образцов
        verification = detection / len(os.listdir(VERIF_IMG_DIR_PATH))
        verified = verification > verification_threshold

        self.verification_label.text = ("Подтверждено" if verified else "Не подтверждено")

        # Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        return results, verified


if __name__ == "__main__":
    CamApp().run()
