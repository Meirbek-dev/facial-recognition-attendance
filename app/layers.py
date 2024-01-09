# Модуль слоя дистанции L1
# Необходим для загрузки модели

import tensorflow as tf
from tensorflow.keras.layers import Layer


@tf.keras.utils.register_keras_serializable(name="ConcatLayer")
class ConcatLayer(Layer):
    def __init__(self, axis=0, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.concat(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


# Сиамский класс дистанции L1
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    # Вычисление схожести
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
