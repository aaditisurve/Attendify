import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPool2D, Flatten, Dense
import pickle


class FaceRecognitionModel:
    def __init__(self, train_data_path, test_data_path, model_save_path='RecognitionModel.h5', map_save_path='StudentsMap.pkl'):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.model_save_path = model_save_path
        self.map_save_path = map_save_path

        self.train_datagen = ImageDataGenerator(rescale=1./255,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True)

        self.test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_set = self.train_datagen.flow_from_directory(self.train_data_path,
                                                                target_size=(64, 64),
                                                                color_mode='grayscale',
                                                                batch_size=32,
                                                                class_mode='categorical')

        self.test_set = self.test_datagen.flow_from_directory(self.test_data_path,
                                                              target_size=(64, 64),
                                                              color_mode='grayscale',
                                                              batch_size=32,
                                                              class_mode='categorical')

        self.TrainClasses = self.train_set.class_indices

        self.ResultMap = {}
        for faceValue, faceName in zip(self.TrainClasses.values(), self.TrainClasses.keys()):
            self.ResultMap[faceValue] = faceName

        # Saving the face map for future reference
        with open(self.map_save_path, 'wb') as fileWriteStream:
            pickle.dump(self.ResultMap, fileWriteStream)

        # The model will give answer as a numeric tag
        # This mapping will help to get the corresponding face name for it

        # The number of neurons for the output layer is equal to the number of faces
        self.OutputNeurons = len(self.ResultMap)

        # Initializing the Convolutional Neural Network
        self.classifier = Sequential()

        # Adding the first layer of CNN
        self.classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64, 64, 1), activation='relu'))

        # MAX Pooling
        self.classifier.add(MaxPool2D(pool_size=(2, 2)))

        # Additional Layer of Convolution for better accuracy
        self.classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        self.classifier.add(MaxPool2D(pool_size=(2, 2)))

        # Flattening
        self.classifier.add(Flatten())

        # Fully Connected Neural Network
        self.classifier.add(Dense(64, activation='relu'))
        self.classifier.add(Dense(self.OutputNeurons, activation='softmax'))

        # Compiling the CNN
        self.classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    def train_model(self, epochs=10):
        self.classifier.fit(self.train_set,
                        epochs=epochs,
                        validation_data=self.test_set)
    
    # Save the trained model
        self.classifier.save('RecognitionModel.keras')
        self.ResultMap = {v: k for k, v in self.train_set.class_indices.items()}  # Update the mapping
        with open(self.map_save_path, 'wb') as fileWriteStream:
            pickle.dump(self.ResultMap, fileWriteStream)

    def predict_image(self, image_path):
        saved_model_path = self.model_save_path
        loaded_model = load_model(saved_model_path)

        test_image = image.load_img(image_path, target_size=(64, 64), color_mode='grayscale')
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        result = loaded_model.predict(test_image, verbose=0)
        with open(self.map_save_path, 'rb') as file:
            # Load the object from the file
            loaded_object = pickle.load(file)
        return loaded_object[np.argmax(result)]


# model = FaceRecognitionModel(train_data_path='Students', test_data_path='Students')
# model.train_model(epochs=10)

#     image_path_to_predict = 'E21CSEU0130.292.jpg'
#     predicted_name = model.predict_image(image_path_to_predict)
#     print(f"The predicted name is: {predicted_name}")
