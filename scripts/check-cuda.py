import tensorflow as tf
import keras 

print("TensorFlow version: ", tf.__version__)
print("Keras version: ", keras.__version__)

GPUS = tf.config.list_physical_devices('GPU') 

if GPUS: 
    CUDA_VERSION = tf.sysconfig.get_build_info()['cuda_version'] 
    print(f"GPU & CUDA {CUDA_VERSION} are available TensorFlow") 
    try: 
        print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"]) 
    except RuntimeError as e: 
        print(e) 
    except KeyError: 
        print("Unable to determine cuDNN version from TensorFlow.") 
else: 
    print("CUDA is not available. TensorFlow does not have CUDA support.")