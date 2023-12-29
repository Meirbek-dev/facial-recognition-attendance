# facial-recognition-attendance

## Разработка информационной системы мониторинга посещаемости на основе распознавания лиц

## [**Полная инструкция установки инструментария**](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)

### Используемые технологии

- **WSL 2 (Ubuntu 22.04)**
- [**Conda** 4.12.0](https://www.anaconda.com/) с [**Python** 3.11.7](https://www.python.org/)
- [**Tensorflow** 2.15.0](https://www.tensorflow.org/)
- [**Keras** 2.15.0](https://keras.io/)
- [**OpenCV** 4.8.1](https://opencv.org/)
- [Предварительно тренированные модели Tensorflow](https://github.com/tensorflow/models)
- [**Protocol Buffers** 3.19.4](https://github.com/protocolbuffers/protobuf/)
- [**SSD MobileNet V2 FPNLite 320x320**](https://github.com/tensorflow/models/blob/master/research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config). [Скачать](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz)
- [**LabelImg**](https://github.com/tzutalin/labelImg). Для графической отметки.
- [**CUDA** 12.2](https://developer.nvidia.com/cuda-11.2.2-download-archive)
- [**cuDNN** 8.9.4](https://developer.nvidia.com/rdp/cudnn-archive)

### [Gist с настройкой WSL](https://gist.github.com/Meirbek-dev/f556979f139ec4a3e346026a9e0246ef)

### [Инструкция по установке и настройке окружения для распознавания объектов при помощи Tensorflow](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)

### [Список GPU, поддерживающих CUDA и оценка их вычислительных возможностей](https://developer.nvidia.com/cuda-gpus)

### [Совместимость версий Tensorflow + Python + Компилятора + Инструмента Сборки + cuDNN + CUDA](https://www.tensorflow.org/install/source#gpu_support_2)

### [Инструкция установки CUDA](https://docs.nvidia.com/cuda/archive/11.2.2/cuda-installation-guide-microsoft-windows/index.html)

### Использованный набор данных

- [**Kaggle**. dataset name](https://www.kaggle.com/link) из *num* изображений
