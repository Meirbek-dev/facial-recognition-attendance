import csv
import json
import logging
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from deepface import DeepFace

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
    img = Image.open(file_path)
    img = img.resize((105, 105))
    img_array = np.array(img) / 255.0
    return img_array


def draw_rectangle_around_face(frame):
    try:
        faces = DeepFace.extract_faces(frame, detector_backend="ssd", enforce_detection=False)
        face = faces[0]
    except ValueError as e:
        logger.error(e)
        return ValueError
    x, y, w, h = map(int, (
        face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']))
    cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)


def draw_rectangles_around_faces(frame, detector):
    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Обнаружение лиц в кадре
    faces = detector(gray)
    try:
        face = faces[0]
    except IndexError:
        return
    
    # Извлечение ограничивающей рамки лица
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    
    # Проверка, что область лица находится в пределах границ кадра
    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0] and w > 0 and h > 0:
        # Обрезание области лица
        face_roi = frame[y:y + h, x:x + w]
        
        if face_roi.size != 0:
            # Обрисовка прямоугольника вокруг обнаруженного лица
            cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
