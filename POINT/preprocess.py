import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from collections import defaultdict

class Preprocess():
    """preprocess the data into vectors
    """      
    def __init__(self, folder_path, max_faces: int = 50) -> None:
        """initialize the class

        Args:
            folder_path (string): the path for the data
            max_faces (int, optional): max number of faces to consider. Defaults to 50.
        """        
        self.folder_path = folder_path
        self.max_faces = max_faces
        
        # define the base model and sub model by calling resnet50 model and taking the last Conv block as output
        self.preprocess_input = tf.keras.applications.resnet50.preprocess_input
        self.base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.sub_model = tf.keras.Model(inputs=self.base_model.input, outputs=self.base_model.layers[-2].output)
        self.sub_model.trainable = False

    def __repr__(self):
        return repr(f'processing {self.folder_path} for embedding extraction with max faces = {self.max_faces}')

    def get_summary(self) -> None:
        """get the summary of the sub model

        Returns:
            None: the sub model summary
        """        
        return self.sub_model.summary()
    
    def create_data(self) -> None:
        """This is an implementation of preprocessing the 
           Multi-scene Important People Image Dataset [https://uoe-my.sharepoint.com/:u:/g/personal/s1798461_ed_ac_uk/EYNSnmbtG2dPuP_ut-Sf2C0Boa7A9vnytWWb11jRiUS-ww?e=iUphhC]
           as follows:
           coordinate images: process convolution layers to get (224, 224, 3) -> (1, 7, 7, 256)
           face images: process through Resnet50 to get (224, 224, 3) -> (1, 7, 7, 2048)
           facecont images: process through Resnet50 to get (224, 224, 3) -> (1, 7, 7, 2048)
           full images: process through Resnet50 to get (224, 224, 3) -> (1, 7, 7, 2048)

           concat coordinate, face images and facecont images to (7, 7, 4352).
           process the concat through convolutional layers to get (7, 7, 4352) -> (1024,)
           process the full images through convolutional layers to get (1, 7, 7, 2048) -> (1024,)

        """        
        self.data = defaultdict(dict)
        self.concat_vector = defaultdict(dict)
        self.image_vector = defaultdict(dict)
        self.labels_vector = defaultdict(dict)
        
        valid_folders = ['Coordinate', 'Face', 'FaceCont', 'Image']

        folder_list = os.listdir(self.folder_path)

        for i ,folder in tqdm(enumerate(folder_list)):
            if folder == '.ipynb_checkpoints': # we only process the structure inside valid folders
              continue
            image_coordinate, image_face, image_facecont, image_full = {}, {}, {}, {}
            label = [0]*self.max_faces
            image_name = folder[-4:]
            sub_folder_path = self.folder_path + os.sep + folder
            for afile in os.listdir(sub_folder_path + os.sep + valid_folders[0]):
                label[int(afile.split('_')[-3])] = int(afile.split('_')[-1][0])
                image_coordinate[afile] = self.process_corr(sub_folder_path + os.sep + valid_folders[0] + os.sep + afile)[0]
            for afile in os.listdir(sub_folder_path + os.sep + valid_folders[1]):
                image_face[afile] = self.sub_model.predict(self.preprocess_image(sub_folder_path + os.sep + valid_folders[1] + os.sep + afile))[0]
            for afile in os.listdir(sub_folder_path + os.sep + valid_folders[2]):
                image_facecont[afile] = self.sub_model.predict(self.preprocess_image(sub_folder_path + os.sep + valid_folders[2] + os.sep + afile))[0]
            for afile in os.listdir(sub_folder_path + os.sep + valid_folders[3]):
                image_full[afile] = self.sub_model.predict(self.preprocess_image(sub_folder_path + os.sep + valid_folders[3] + os.sep + afile))[0]
                break # all the images are the same in the Image folder so we need to process only one of them

            self.data.update({image_name: {'Coordinate': image_coordinate, 
                                           'Face': image_face, 
                                           'FaceCont': image_facecont, 
                                           'Image': image_full,
                                           'label': label}})

            num_faces = len(self.data[image_name]['Coordinate'])
            # if num_faces > self.max_faces: self.max_faces = num_faces
            for face_num in range(num_faces):
                prefix = f'Image_{image_name}_'
                suffix = f'{"{0:0=2d}".format(face_num)}'

                coor_prefix = prefix + 'Coor_' + suffix
                face_prefix = prefix + 'Face_' + suffix

                concat_vec = np.concatenate([self.data[image_name]['Coordinate'][[k for k in self.data[image_name]['Coordinate'].keys() if coor_prefix in k][0]], 
                                            self.data[image_name]['Face'][[k for k in self.data[image_name]['Face'].keys() if face_prefix in k][0]], 
                                            self.data[image_name]['FaceCont'][[k for k in self.data[image_name]['FaceCont'].keys() if face_prefix in k][0]]
                                            ], -1) # (7, 7, 4352)

                dense_from_concat_vec = self.process_conv(concat_vec) # (1024,)
                self.concat_vector[image_name].update({suffix: dense_from_concat_vec})
            if len(self.concat_vector[image_name]) < self.max_faces:
                for i in range(self.max_faces - len(self.concat_vector[image_name])):
                    self.concat_vector[image_name].update({f'tmp_{i}': tf.zeros_like(np.arange(1024, dtype=float))})

            image_prefix = prefix + 'Img_'
            dense_from_full_image_vec = self.process_conv(self.data[image_name]['Image'][[k for k in self.data[image_name]['Image'].keys() if image_prefix in k][0]])
            self.image_vector[image_name] = dense_from_full_image_vec

    def preprocess_image(self, filepath: str) -> np.ndarray:
        '''perform decoding and resizing to an image
        Args:
            filepath (string): filepath of an image
        Returns:
            tf.Tensor: the image after decoding and resizing
        '''        
        image = tf.io.read_file(filename=filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224,224], method='nearest')

        return self.preprocess_input(np.expand_dims(image, axis=0))
    
    def process_corr(self, filepath: str) -> tf.Tensor:
        """process image throgh five convolutional layers and maxpooling to get (1, 7, 7, 256) output

        Args:
            filepath (str): the filepath for the image

        Returns:
            tf.Tensor: the tensor after convolutional layers
        """        

        image = self.preprocess_image(filepath)
        input_shape = image.shape # (224, 224, 3)
        
        initializer = tf.keras.initializers.GlorotNormal(seed=42)

        x = tf.keras.layers.Conv2D(10, 1, activation='relu', input_shape=input_shape, kernel_initializer=initializer)(image) # (1, 112, 112, 10)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)
        
        x = tf.keras.layers.Conv2D(32, 1, activation='relu', input_shape=x.shape, kernel_initializer=initializer)(x) # (1, 56, 56, 32)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)
        
        x = tf.keras.layers.Conv2D(64, 1, activation='relu', input_shape=x.shape, kernel_initializer=initializer)(x) # (1, 28, 28, 64)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)
        
        x = tf.keras.layers.Conv2D(128, 1, activation='relu', input_shape=x.shape, kernel_initializer=initializer)(x) # (1, 14, 14, 128)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)
        
        x = tf.keras.layers.Conv2D(256, 1, activation='relu', input_shape=x.shape, kernel_initializer=initializer)(x) # (1, 7, 7, 256)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)

        return x
      
    def process_conv(self, conv) -> tf.Tensor:
        """process concat vector tensor throgh two convolutional layers and dense layer to get (1, 1, 1, 1024) output

        Args:
            conv (np.array): concat vector

        Returns:
            tf.Tensor: the tensor after convolutional layers and dense layer
        """        
        
        input_image = conv.reshape((-1, ) + conv.shape)
        
        initializer = tf.keras.initializers.GlorotNormal(seed=42)

        x = tf.keras.layers.Conv2D(4352/2 + 4352/4, 1, activation='relu', input_shape=input_image.shape, kernel_initializer=initializer)(input_image) # (1, 7, 7, 4352)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)

        x = tf.keras.layers.Conv2D(4352/2, 1, activation='relu', input_shape=x.shape, kernel_initializer=initializer)(x) # (1, 3, 3, 2176)
        x = tf.keras.layers.MaxPool2D(2, 2)(x) # (1, 1, 1, 1024)

        x = tf.keras.layers.Dense(1024, kernel_initializer=initializer)(x)

        return x[0][0][0]