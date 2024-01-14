# Модуль слоя дистанции L1
# Необходим для загрузки модели

import tensorflow as tf
from tensorflow.keras.layers import Layer


# Сиамский класс дистанции L1
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    
    # Вычисление схожести
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
