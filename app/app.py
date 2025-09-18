from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow_hub as hub

app = Flask(__name__)

# Load model components
with open("model_components/char_vectorizer.pkl", "rb") as f:
    char_vectorizer = pickle.load(f)

path = "universal-sentence-encoder/4"
tf_hub_embedding_layer = hub.KerasLayer(path, input_shape=[], dtype=tf.string, trainable=False)

class Embedding(Layer):
    def call(self, x):
        return tf_hub_embedding_layer(x)
    

final_model_path = "model_components/final_model.keras"
final_model = laod_model(
    final_model_path,
    custom_objects={
        "TextVectorization": char_vectorizer,
        "Embedding": Embedding,
        "KerasLayer": hub.KerasLayer
    }
)

# Load label encoder