sudo apt-get update && sudo apt-get upgrade -y
sudo apt autoremove -y
touch /home/meirb/.hushlogin
export CC=clang
export CXX=clang++
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh 
rm Anaconda3-2023.09-0-Linux-x86_64.sh 
conda install scikit-learn-intelex
conda update conda
conda update --all
conda install pip
which pip
sudo apt-key del 7fa2af80
sudo ls /etc/apt/trusted.gpg.d
sudo rm /etc/apt/trusted.gpg.d/7fa2af80.gpg
sudo nano /etc/wsl.conf
sudo nano /etc/hosts

conda create -n frs pip python=3.11
sudo apt install python3-dev python3-venv gcc clang build-essential xz-utils curl -y
sudo apt-get --with-new-pkgs upgrade python3-update-manager update-manager-core
conda activate frs
python3 -m pip install tensorflow[and-cuda]
sudo update-pciids
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

nvidia-smi
lspci | grep -i nvidia

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'TENSORRT_PATH=$(dirname $(python -c "import tensorrt;print(tensorrt.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
python check-cuda.py 
pip install tensorrt
conda activate frs
python check-cuda.py 

mkdir crash-reports
export MLIR_CRASH_REPRODUCER_DIRECTORY=/home/meirb/crash-reports
