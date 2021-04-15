import tensorflow as tf
from tensorflow.keras import metrics
import numpy as np

def get_pesq(ref_sig, out_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ: numpy.ndarray, [B]
    """
    return np.array([
        pesq(sr, ref, out, 'wb') 
        for ref, out in zip(ref_sig, out_sig)])


def get_stoi(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI: numpy.ndarray, [B]
    """
    return np.array([
        stoi(ref, out, sr, extended=False) 
        for ref, out in zip(ref_sig, out_sig)])

class PESQ(metrics.Metric):
    def __init__(self, sample_rate=16_000, name='PESQ', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum = self.add_weight(name='sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.sample_rate = sample_rate

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reduce_mean(y_true, axis=-1).numpy()
        y_pred = tf.reduce_mean(y_pred, axis=-1).numpy()

        pesq = get_pesq(y_true, y_pred, sr=self.sample_rate)
        if sample_weight is not None:
            pesq *= sample_weight.numpy()
        
        self.sum.assign_add(np.sum(pesq))
        self.count.assign_add(len(pesq))

    def result(self):
        return self.sum / self.count

class STOI(metrics.Metric):
    def __init__(self, sample_rate=16_000, name='STOI', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum = self.add_weight(name='sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.sample_rate = sample_rate

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reduce_mean(y_true, axis=-1).numpy()
        y_pred = tf.reduce_mean(y_pred, axis=-1).numpy()

        stoi = get_stoi(y_true, y_pred, sr=self.sample_rate)
        if sample_weight is not None:
            stoi *= sample_weight.numpy()
        
        self.sum.assign_add(np.sum(stoi))
        self.count.assign_add(len(stoi))

    def result(self):
        return self.sum / self.count