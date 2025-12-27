import tensorflow as tf
from tensorflow import keras
import numpy as np

# 自定義 Layer Normalization 層
class CustomLayerNormalization(keras.layers.Layer):
    def __init__(self, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # 定義兩個可訓練權重 alpha 和 beta，形狀為 input_shape[-1:]
        # alpha 初始化為全1，beta 初始化為全0
        self.alpha = self.add_weight(
            name="alpha",
            shape=input_shape[-1:],
            initializer="ones",
            dtype=tf.float32
        )
        self.beta = self.add_weight(
            name="beta",
            shape=input_shape[-1:],
            initializer="zeros",
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, inputs):
        # 計算每個實例的均值和方差，axes=-1 表示在最後一個軸上計算（特徵軸）
        mean, variance = tf.nn.moments(inputs, axes=-1, keepdims=True)
        # 標準差 = 方差的平方根
        std = tf.sqrt(variance)
        # 計算歸一化值：alpha ⊙ (X - μ) / (σ + ε) + beta
        normalized = (inputs - mean) / (std + self.epsilon)
        return self.alpha * normalized + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

# 測試自定義層與 Keras 官方層的輸出一致性
def test_layer_norm():
    # 生成隨機輸入數據
    batch_size, seq_len, features = 2, 3, 4
    np.random.seed(42)
    X = np.random.randn(batch_size, seq_len, features).astype(np.float32)

    # 使用自定義層
    custom_ln = CustomLayerNormalization(epsilon=1e-3)
    custom_output = custom_ln(X)

    # 使用 Keras 官方層，注意設定 epsilon 相同
    keras_ln = keras.layers.LayerNormalization(epsilon=1e-3)
    keras_output = keras_ln(X)

    # 比較輸出
    print("自定義層輸出形狀:", custom_output.shape)
    print("Keras 官方層輸出形狀:", keras_output.shape)
    print("輸出差值最大絕對值:", np.max(np.abs(custom_output - keras_output)))
    print("輸出是否接近:", np.allclose(custom_output, keras_output, atol=1e-4))

    # 也可以檢查權重是否可訓練
    print("\n自定義層 alpha 權重:", custom_ln.alpha.numpy())
    print("自定義層 beta 權重:", custom_ln.beta.numpy())
    print("Keras 官方層 gamma 權重:", keras_ln.gamma.numpy())
    print("Keras 官方層 beta 權重:", keras_ln.beta.numpy())

if __name__ == "__main__":
    test_layer_norm()