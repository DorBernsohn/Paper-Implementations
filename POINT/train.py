import os
import numpy as np
from attention import *
import tensorflow as tf
from typing import Tuple
from utils import set_seed
from preprocess import Preprocess
from tensorflow.keras import regularizers

set_seed(42)

def create_model(drop_out_rate: float = 0.3, learning_rate: float = 0.00001, max_faces: int = 50) -> tf.keras.Model:
    """create_model create the model

    Args:
        drop_out_rate (float, optional): the drop out rate. Defaults to 0.3.
        learning_rate (float, optional): the learning rate. Defaults to 0.00001.
        max_faces (int, optional): the max number of faces. Defaults to 50.

    Returns:
        tf.keras.Model: the model for training
    """    
    interior_input = tf.keras.layers.Input(shape=(max_faces, 2048,), dtype="float32")

    context_vector, attention_weights = MultiHeadAttention(d_model=512, num_heads=8)(interior_input, interior_input, interior_input, None)
    dense = tf.keras.layers.Dense(512, activation="relu", 
                kernel_regularizer=regularizers.l2(0.01))(context_vector)
    dropout = tf.keras.layers.Dropout(drop_out_rate)(dense)
    dense = tf.keras.layers.Dense(256, activation="relu")(dropout)
    dropout = tf.keras.layers.Dropout(drop_out_rate)(dense)
    dense = tf.keras.layers.Dense(128, activation="relu")(dropout)
    dropout = tf.keras.layers.Dropout(drop_out_rate)(dense)
    dense = tf.keras.layers.Dense(64, activation="relu")(dropout)
    dropout = tf.keras.layers.Dropout(drop_out_rate)(dense)
    dense = tf.keras.layers.Dense(32, activation="relu")(dropout)
    dropout = tf.keras.layers.Dropout(drop_out_rate)(dense)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(dropout)
    
    model = tf.keras.Model(inputs=interior_input, outputs=output)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model

def scheduler(epoch: int, lr: float) -> float:
    """scheduler learning rate scheduler callback

    Args:
        epoch (int): the currnet epoch number
        lr (float): the current learning rate

    Returns:
        float: the new learning rate
    """    
    if epoch < 20:
        return lr
    else:
        if epoch % 20 == 0:
            return lr / 10
        else:
            return lr

def create_data(preprocess_dict: dict) -> Tuple[np.array, np.array]:
    """create_data create x, y to the model

    Args:
        preprocess_dict (dict): the preprocess dictionary of the data

    Returns:
        [np.array, np.array]: the x and y to the model
    """    
    x = []
    y = []
    for i, v in enumerate(preprocess_dict.concat_vector.keys()):
        tmp = []
        for k in preprocess_dict.concat_vector[v]:
            tmp.append(preprocess_dict.concat_vector[v][k].numpy())
        tmp = [np.concatenate([preprocess_dict.image_vector[v].numpy(), x]) for x in tmp]
        x.append(np.array(tmp))
        y.append(np.array(preprocess_dict.data[v]['label']))
    return x, y

if __name__ == '__main__':
    preprocess_dict = Preprocess(r"/POINT/data/MSDataSet_process")
    preprocess_dict.create_data()

    model = create_model(max_faces=preprocess_dict.max_faces)
    print(model.summary())
    x, y = create_data(preprocess_dict)

    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
    
    model.fit(np.array(x), 
          np.array(y),
          validation_split=0.1,
          epochs = 100,
          batch_size=2,
          callbacks = [early_stopping_callback])
    if 'saved_model' not in os.listdir():
        os.system('mkdir -p saved_model')
    model.save('saved_model/my_model')