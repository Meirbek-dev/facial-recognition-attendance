import csv
from datetime import datetime

import tensorflow as tf

KZ_DATETIME_FMT = "%d.%m.%Y %H:%M:%S"


def register_attendance(name, faculty, group, file_path):
    try:
        with open(file_path, newline='', encoding='utf-8') as file:
            # Проверка существования файла
            first_char = file.read(1)
            file.seek(0)  # Сброс указателя файла на начало

            # Запись заголовка, если файл пуст
            if not first_char:
                header = ['ID', 'Name', 'Faculty', 'Group', 'AttendanceTime']
                writer = csv.writer(file)
                writer.writerow(header)
    except FileNotFoundError:
        # Создание файла с заголовком, в случае если файл не существует
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            header = ['ID', 'Name', 'Faculty', 'Group', 'AttendanceTime']
            writer = csv.writer(file)
            writer.writerow(header)

    # Получение текущей даты и времени
    current_datetime = datetime.now()
    attendance_time = current_datetime.strftime(KZ_DATETIME_FMT)

    # Считывание существующих ID
    existing_ids = set()
    try:
        with open(file_path, newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Пропуск заголовков
            existing_ids = {int(row[0]) for row in reader}
    except FileNotFoundError:
        pass

    # Сгенерировать новый ID
    new_id = 1
    while new_id in existing_ids:
        new_id += 1

    # Запись данных в csv-файл
    data = [new_id, name, faculty, group, attendance_time]
    with open(file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data)


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
