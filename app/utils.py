import csv
import json
import logging
import os
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from deepface.DeepFace import extract_faces

from layers import L1Dist

DATETIME_FMT = "%d.%m.%Y %H:%M:%S"
ENCODING = "utf-8"
HEADER = ["UserID", "FirstName", "LastName", "Faculty", "Group", "AttendanceTime"]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_json(file_path):
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


def draw_rectangle_around_face(frame):
    try:
        faces = extract_faces(frame, detector_backend="yolov8", enforce_detection=False)
        face = faces[0]
    except ValueError as e:
        logger.error(e)
        return ValueError
    x, y, w, h = map(
        int,
        (
            face["facial_area"]["x"],
            face["facial_area"]["y"],
            face["facial_area"]["w"],
            face["facial_area"]["h"],
        ),
    )
    cv2.rectangle(frame, (x, y), (x + 10 + w + 10, y + 10 + h + 10), (0, 255, 0), 2)


def ensure_csv_file_exists(file_path, header):
    with open(file_path, "a+", newline="", encoding=ENCODING) as file:
        file.seek(0)
        existing_content = file.read()
        if not existing_content or not any(item in existing_content for item in header):
            file.seek(0)
            csv.writer(file).writerow(header)


def read_csv(file_path):
    try:
        with open(file_path, newline="", encoding=ENCODING) as file:
            return list(csv.DictReader(file))
    except FileNotFoundError:
        return []


def write_rows_to_csv(file_path, header, data):
    with open(file_path, "a", newline="", encoding=ENCODING) as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if file.tell() == 0:
            # Создание файла с заголовком, в случае если файл не существует
            writer.writeheader()
        writer.writerows(data)


def get_user_data(user_id):
    return load_json(
        os.path.join("app_data", "verification_data", user_id, "info.json")
    )


def register_attendance(user_id, file_path):
    ensure_csv_file_exists(file_path, HEADER)
    user_data = get_user_data(user_id)
    user_data = {
        "user_id": user_data.get("user_id", ""),
        "first_name": user_data.get("first_name", ""),
        "last_name": user_data.get("last_name", ""),
        "email": user_data.get("email", ""),
        "faculty": user_data.get("faculty", ""),
        "group": user_data.get("group", ""),
        "address": user_data.get("address", ""),
    }

    assert (
            user_id == user_data["user_id"]
    ), "Пользователь с данным идентификатором не существует"

    # Получение текущей даты и времени
    attendance_time = datetime.now().strftime(DATETIME_FMT)

    # Запись данных в csv-файл
    new_row = {
        "UserID": user_id,
        "FirstName": user_data["first_name"],
        "LastName": user_data["last_name"],
        "Faculty": user_data["faculty"],
        "Group": user_data["group"],
        "AttendanceTime": attendance_time,
    }
    write_rows_to_csv(file_path, HEADER, [new_row])
    logger.info(f"Подтврежден {new_row}. ")
    return user_data


def load_model(path):
    try:
        return tf.keras.models.load_model(path, custom_objects={"L1Dist": L1Dist})
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        return None


# Загрузка изображения из файла и конвертация в 105x105px
def preprocess(file_path):
    img = Image.open(file_path)
    img = img.resize((105, 105))
    img_array = np.array(img) / 255.0
    return img_array
