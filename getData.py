import seaborn as sns
from metaClasses import *
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from trainingDynamics import TrainingDynamicsCallback
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, glue_convert_examples_to_features

class getData(Structures):
    # ['rte', 'qqp', 'mrpc', 'cola', 'qnli']
    _fields = ['dataset_name', 'model_name', 'max_length', 'learning_rate', 'epochs']
    _dataset_name = String('dataset_name')
    _model_name = String('model_name')
    _max_length = Integer('max_length')
    _learning_rate = Float('learning_rate')
    _epochs = Integer('epochs')

    def load_data(self):
        train = tfds.load('glue/' + self.dataset_name, split='train', shuffle_files=True)
        train_unshuffled = tfds.load('glue/' + self.dataset_name, split='train', shuffle_files=False)
        validation = tfds.load('glue/' + self.dataset_name, split='validation', shuffle_files=True)
        # test = tfds.load('glue/' + self.dataset_name, split='test', shuffle_files=True)

        # Prepare datasets for Huggingface's transformers
        tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        train = glue_convert_examples_to_features(train, tokenizer, max_length=self.max_length, task=self.dataset_name)
        self.train_unshuffled = glue_convert_examples_to_features(train_unshuffled, tokenizer, max_length=self.max_length, task=self.dataset_name)
        validation = glue_convert_examples_to_features(validation, tokenizer, max_length=self.max_length, task=self.dataset_name)
        # test = glue_convert_examples_to_features(test, tokenizer, max_length=self.max_length, task=self.dataset_name)

        self.validation = validation.batch(self.max_length).prefetch(1)
        self.train = train.shuffle(1000).repeat().batch(int(self.max_length/2)).prefetch(1)

    def train_model(self):
        self.model = TFDistilBertForSequenceClassification.from_pretrained(self.model_name)
        self.dataset_map = TrainingDynamicsCallback(self.train_unshuffled.batch(2*self.max_length), 
                                           outputs_mapping_probabilities=lambda x: tf.nn.softmax(x[0]),
                                           sparse_labels=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        callbacks = [self.dataset_map, tf.keras.callbacks.EarlyStopping(patience=3)]

        self.model.compile(optimizer=optimizer, loss=loss)

        self.model.fit(self.train, epochs=self.epochs, validation_data=self.validation, steps_per_epoch=150, callbacks=callbacks)
    
    def kde_plot(self):
        _, ax = plt.subplots(figsize=(9, 7))


        sns.scatterplot(x=self.dataset_map.variability, y=self.dataset_map.confidence, hue=self.dataset_map.correctness, ax=ax)
        sns.kdeplot(x=self.dataset_map.variability, y=self.dataset_map.confidence, 
                    levels=8, color=sns.color_palette("Paired")[7], linewidths=1, ax=ax)

        ax.set(title=f'DataSet: {self.dataset_name} (train set)\nModel: {self.model_name} classifier',
            xlabel='Variability', ylabel='Confidence')

        ax.legend(title='Correctness')
        ax.figure.savefig(f'{self.dataset_name}.png')