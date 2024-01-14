import csv
import json
from datetime import datetime

import tensorflow as tf

DATETIME_FMT = "%d.%m.%Y %H:%M:%S"
ENCODING = "utf-8"
HEADER = ['ID', 'Name', 'Faculty', 'Group', 'AttendanceTime']


def get_config(file_path):
    with open(file_path, encoding=ENCODING) as json_file:
        return json.load(json_file)


def ensure_file_exists(file_path, header):
    with open(file_path, "a+", newline='', encoding=ENCODING) as file:
        file.seek(0)
        existing_content = file.read()
        if not existing_content or not any(item in existing_content for item in header):
            file.seek(0)
            csv.writer(file).writerow(header)


# Считывание существующих ID
def get_existing_ids(file_path):
    try:
        with open(file_path, newline='', encoding=ENCODING) as file:
            reader = csv.DictReader(file)
            return {int(row['ID']) for row in reader if 'ID' in row and row['ID'].isdigit()}
    except FileNotFoundError:
        return set()


# Генерация нового ID
def generate_new_id(existing_ids):
    return max(existing_ids, default=0) + 1


def write_row_to_csv(file_path, header, data):
    with open(file_path, 'a', newline='', encoding=ENCODING) as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if file.tell() == 0:
            # Создание файла с заголовком, в случае если файл не существует
            writer.writeheader()
        writer.writerow(dict(zip(header, data)))


def register_attendance(name, faculty, group, file_path):
    ensure_file_exists(file_path, HEADER)
    # Получение текущей даты и времени
    attendance_time = datetime.now().strftime(DATETIME_FMT)
    
    existing_ids = get_existing_ids(file_path)
    new_id = generate_new_id(existing_ids)
    
    # Запись данных в csv-файл
    data = [new_id, name, faculty, group, attendance_time]
    write_row_to_csv(file_path, HEADER, data)


# Загрузка изображения из файла и конвертация в 105x105px
def preprocess(file_path):
    # Чтение и загрузка изображения
    img = tf.io.decode_image(tf.io.read_file(file_path))
    # Изменение размера изображения на 105x105
    img = tf.image.resize(img, (105, 105))
    # Масштабирование изображения в диапазоне от 0 до 1
    img /= 255.0
    return img
