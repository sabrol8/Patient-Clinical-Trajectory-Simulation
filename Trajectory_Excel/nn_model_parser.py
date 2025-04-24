import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


class ClusterPredictor:
    ''' A simple class to easily query the neural networks from inside AnyLogic using Pypeline '''
    def __init__(self):
        # load the model
        self.trajectory = load_model("predict_trajectory.h5")

    def predict_cluster(self, dynamic, static):
        # convert default list to numpy array
        darray = np.array(dynamic)
        sarray = np.array(static)
        # query the neural network for the prediction
        prediction = self.trajectory.predict([darray, sarray])
        probabilities = tf.nn.softmax(prediction[0]).numpy()
        
        # Find the index where the probability is maximum
        max_index = np.argmax(probabilities)
        
        return max_index