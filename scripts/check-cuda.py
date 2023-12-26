import tensorflow as tf

print("TensorFlow version: ", tf.version.VERSION)
GPUS = tf.config.list_physical_devices('GPU')
# Check for GPU, CUDA & cuDNN availability
if GPUS:
    CUDA_VERSION = tf.sysconfig.get_build_info()['cuda_version']
    print("CUDA is available. TensorFlow is built with CUDA", CUDA_VERSION, "support.")
    try:
        # Set memory growth for visible GPUs
        for gpu in GPUS:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs are visible to TensorFlow and set for memory growth.")
        # Check for cuDNN compatibility directly from TensorFlow
        print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    except KeyError:
        print("Unable to determine cuDNN version from TensorFlow.")
else:
    print("CUDA is not available. TensorFlow is not built with CUDA support.")
