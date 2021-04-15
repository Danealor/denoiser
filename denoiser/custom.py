import tensorflow as tf
from tensorflow.keras import layers, losses

class BLSTM(layers.Layer):
    def __init__(self, num_layers=2, bi=True, stateful=False, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.bidirectional = bi
        self.stateful = stateful

    def get_config(self):
        config = super().get_config()
        config.update({"num_layers": self.num_layers})
        config.update({"bidirectional": self.bidirectional})
        config.update({"stateful": self.stateful})
        return config

    # This might be more efficient if implemented using cudNN LSTM
    def _stacked_lstm_impl1(self, dim):
        lstm_layers = []
        for layer in range(self.num_layers):
            lstm = layers.LSTM(dim, return_sequences=True, stateful=self.stateful)
            if self.bidirectional:
                lstm = layers.Bidirectional(lstm)
            lstm_layers.append(lstm)
        return lstm_layers

    def _stacked_lstm_impl2(self, dim):
        rnn_cells = [layers.LSTMCell(dim) for _ in range(self.num_layers)]
        stacked_lstm = layers.StackedRNNCells(rnn_cells)
        lstm_layer = layers.RNN(stacked_lstm, return_sequences=True, stateful=self.stateful)
        if self.bidirectional:
            lstm_layer = layers.Bidirectional(lstm_layer)
        return [lstm_layer]

    def build(self, input_shape):
        dim = input_shape[-1]
        self.lstm_layers = self._stacked_lstm_impl1(dim)
        if self.bidirectional:
            self.linear = layers.Dense(dim)

    def call(self, inputs):
        x = inputs
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x)
        if self.bidirectional:
            x = self.linear(x)
        return x


class GLU(layers.Layer):
    def __init__(self, dim=-1,**kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config

    def build(self, input_shape):
        if (input_shape[self.dim] % 2 != 0):
            raise ValueError(f"Dimension of splitting (dim = {self.dim}) " + 
                             f"should be divisble by two. " +
                             f"Instead found: {input_shape[self.dim]}")
        half = input_shape[self.dim] // 2
        self.half_size = [-1]*input_shape.ndims
        self.half_size[self.dim] = half
        self.half_start = [0]*input_shape.ndims
        self.half_start[self.dim] = half

    def call(self, inputs):
        first = tf.slice(inputs,[0]*inputs.shape.ndims,self.half_size)
        last = tf.slice(inputs,self.half_start,self.half_size)
        return first * tf.sigmoid(last)


class Normalize(layers.Layer):
    def __init__(self, floor=1e-3,**kwargs):
        super().__init__(**kwargs)
        self.floor = floor

    def get_config(self):
        config = super().get_config()
        config.update({"floor": self.floor})
        return config

    def call(self, inputs):
        mono = tf.math.reduce_mean(inputs, axis=1)
        std = tf.math.reduce_std(mono, axis=-1)
        std += self.floor
        std_unsqueezed = std[...,tf.newaxis,tf.newaxis]
        return inputs / std_unsqueezed, std_unsqueezed

    def norm_denorm(self, x):
        norm, std = self(x)
        def denorm(x):
            return x * std
        return norm, denorm


class SumLoss(losses.Loss):
    def __init__(self, *losses, **kwargs):
        super().__init__(**kwargs)
        self.losses = losses

    def call(self, y_true, y_pred):
        return sum(loss(y_true, y_pred) for loss in self.losses)