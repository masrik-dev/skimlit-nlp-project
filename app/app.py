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
with open("model_components/label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

def classify_abstract_sentences(example_sentence, model):
    abstract_lines = example_sentence.split(". ")
    abstract_lines = [line.strip() for line in abstract_lines if line.strip()]
    total_lines_in_sample = len(abstract_lines)

    sample_lines = []
    for i, line in enumerate(abstract_lines):
        sample_dict = {
            "text": line,
            "line_number": i,
            "total_lines": total_lines_in_sample - 1
        }
        sample_lines.append(sample_dict)

        test_abstract_line_numbers = [line["line_number"] for line in sample_lines]
        test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15)

        test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
        test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)

        