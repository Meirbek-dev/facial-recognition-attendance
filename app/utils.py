import csv
import json
import logging
from datetime import datetime

import cv2
import tensorflow as tf
# from kivy.graphics.texture import Texture
from PIL import Image, ImageTk

from layers import L1Dist

DATETIME_FMT = "%d.%m.%Y %H:%M:%S"
ENCODING = "utf-8"
HEADER = ['ID', 'Name', 'Faculty', 'Group', 'AttendanceTime']

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_config(file_path):
    with open(file_path, encoding=ENCODING) as json_file:
        return json.load(json_file)


def setup_video_capture(video_source):
    try:
        return cv2.VideoCapture(video_source)
    except Exception as e:
        logger.error(f"Ошибка при подключении к устройству видео-захвата: {e}")
        return None


def convert_to_tkinter_image(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    return ImageTk.PhotoImage(image=img)


# def setup_web_cam_texture():
#     texture = Texture.create(size=(1, 1), colorfmt="bgr")
#     texture.flip_vertical()
#     texture.flip_horizontal()
#     return texture


def load_model(path):
    try:
        return tf.keras.models.load_model(path, custom_objects={"L1Dist": L1Dist})
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        return None


def ensure_csv_file_exists(file_path, header):
    with open(file_path, "a+", newline='', encoding=ENCODING) as file:
        file.seek(0)
        existing_content = file.read()
        if not existing_content or not any(item in existing_content for item in header):
            file.seek(0)
            csv.writer(file).writerow(header)


def read_csv(file_path):
    try:
        with open(file_path, newline='', encoding=ENCODING) as file:
            return list(csv.DictReader(file))
    except FileNotFoundError:
        return []


def write_rows_to_csv(file_path, header, data):
    with open(file_path, 'a', newline='', encoding=ENCODING) as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if file.tell() == 0:
            # Создание файла с заголовком, в случае если файл не существует
            writer.writeheader()
        writer.writerows(data)


def register_attendance(name, faculty, group, file_path):
    ensure_csv_file_exists(file_path, HEADER)
    data = read_csv(file_path)
    
    # Получение текущей даты и времени
    attendance_time = datetime.now().strftime(DATETIME_FMT)
    
    new_id = max((int(row['ID']) for row in data), default=0) + 1
    
    # Запись данных в csv-файл
    new_row = {'ID': new_id, 'Name': name, 'Faculty': faculty, 'Group': group, 'AttendanceTime': attendance_time}
    write_rows_to_csv(file_path, HEADER, [new_row])


# Загрузка изображения из файла и конвертация в 105x105px
def preprocess(file_path):
    # Чтение и загрузка изображения
    img = tf.io.decode_image(tf.io.read_file(file_path))
    # Изменение размера изображения на 105x105
    img = tf.image.resize(img, (105, 105))
    # Масштабирование изображения в диапазоне от 0 до 1
    img /= 255.0
    return img
