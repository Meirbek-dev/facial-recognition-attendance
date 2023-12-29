# WSL-SETUP

## Конфигурация Windows Subsystem for Linux (Ubuntu 22.04.3) для разработки на Tensorflow + CUDA

Обновление системы и очистка от старых пакетов

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt autoremove -y
```

Создание .hushlogin

```bash
touch /home/meirb/.hushlogin
```

Изменение названия хоста

```bash
sudo nano /etc/wsl.conf
sudo nano /etc/hosts
```

Установка Anaconda

```bash
LATEST_ANACONDA=$(wget -O - https://www.anaconda.com/distribution/ 2>/dev/null | sed -ne 's@.*\(https:\/\/repo\.anaconda\.com\/archive\/Anaconda3-.*-Linux-x86_64\.sh\)\">64-Bit (x86) Installer.*@\1@p')
wget $LATEST_ANACONDA
chmod +x Anaconda3*.sh
./Anaconda3*.sh
rm Anaconda3-2023.09-0-Linux-x86_64.sh
```

Ускорение scikit-learn

```bash
conda install scikit-learn-intelex
```

Обновление пакетов Anaconda

```bash
conda update conda
conda update --all
```

Установка и проверка pip

```bash
conda install pip
which pip
```

Удаление ключа перед установкой CUDA + cuDNN

```bash
sudo apt-key del 7fa2af80
sudo rm /etc/apt/trusted.gpg.d/7fa2af80.gpg
```

Создание директории для хранения отчета о крашах

```bash
mkdir crash-reprod
export MLIR_CRASH_REPRODUCER_DIRECTORY=/home/meirb/crash-reprod
```

Создание среды для разработки с установкой TensorFlow + CUDA

```bash
conda create -n frs pip python=3.11
conda activate frs
pip install tensorflow[and-cuda] # Стабильная версия
pip install tf-nightly[and-cuda] # Ночная версия
conda install opencv
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))" # Проверка работы CPU
```

Автоматическая активация среды при запуске

```bash
echo "conda activate frs" >> ~/.bashrc
```

Установка и добавление alias для Jupyter Lab & Notebook

```bash
conda install jupyterlab notebook jupyter_contrib_nbextensions nb_conda_kernels LenvsSlidesExporter chardet
echo "alias jl='jupyter-lab'" >> ~/.bashrc
echo "alias jn='jupyter-notebook'" >> ~/.bashrc
source ~/.bashrc
```

Проверка информации о GPU в WSL

```bash
nvidia-smi
```

Отключение отображения сообщений о недоступности NUMA

```bash
export TF_CPP_MIN_LOG_LEVEL=1
```

Пулл официального контейнера tensorflow

```bash
docker pull tensorflow/tensorflow

```

Запуск контейнера GPU, с интерпретатором Python.

```bash
docker run -it --rm --gpus all tensorflow/tensorflow:latest-gpu python
```

Запуск сервера Jupyter Notebook.

```bash
docker run -it --rm --gpus all -v $(realpath ~/notebooks):/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter
```
