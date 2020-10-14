import numpy as np
from tqdm import tqdm
import tensorflow as tf
from metaClasses import *

class TrainingDynamicsCallback(Structures, tf.keras.callbacks.Callback):
    _fields = ['dataset', 'outputs_mapping_probabilities', 'sparse_labels']
    _dataset = TensorFlowDataSet('dataset')
    _outputs_to_probabilities = TensorFlowTensor('outputs_mapping_probabilities')
    _sparse_labels = Bool('sparse_labels')
    _pred_probabilities = None


    @property
    def confidence(self) -> np.ndarray:
        return np.mean(self._pred_probabilities, axis=-1)

    @property
    def variability(self) -> np.ndarray:
        return np.std(self._pred_probabilities, axis=-1)

    @property
    def correctness(self) -> np.ndarray:
        return np.mean(self._pred_probabilities > 0.5, axis=-1)

    
    def on_epoch_end(self, epoch, logs=None):
        # Gather gold label probabilities over the dataset
        pred_probabilities = list()
        for x, y in self.dataset:
            probabilities = self.model.predict(x)

            if self.outputs_mapping_probabilities is not None:
                probabilities = self.outputs_mapping_probabilities(probabilities)

            if self.sparse_labels:
                y = tf.one_hot(y, depth=probabilities.shape[-1])

            if len(y.shape) == 1:
                probabilities = tf.squeeze(probabilities)
                y = tf.squeeze(y)
                batch_probabilities = tf.where(y == 0, 1 - probabilities, probabilities)

            elif len(y.shape) == 2:
                tensor = probabilities
                mask = tf.cast(y, tf.bool)
                batch_probabilities = tf.boolean_mask(tensor, mask).numpy()

            else:
                raise ValueError('Got a y with shape of  {y.shape}.')

            pred_probabilities = np.append(pred_probabilities, [batch_probabilities])

        if self._pred_probabilities is None:  # Happens only on first iteration
            self._pred_probabilities = pred_probabilities[:, np.newaxis]
        else:
            self._pred_probabilities = np.hstack([self._pred_probabilities, pred_probabilities[:, np.newaxis]])