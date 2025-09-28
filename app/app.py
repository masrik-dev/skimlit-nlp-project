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

    def split_chars(text):
        return " ".join(list(text))
        
    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]

    test_abstract_pred_probs = model.predict(x=(test_abstract_line_numbers_one_hot,
                                                test_abstract_total_lines_one_hot,
                                                tf.constant(abstract_lines),
                                                tf.constant(abstract_chars)))
        
    test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)

    test_abstract_pred_classes = [label_encoder.classes_[i] for i in test_abstract_preds]

    output = []
    for i, line in enumerate(abstract_lines):
        output.append(f"{test_abstract_pred_classes[i]}: {line}")
    return output


def preprocess_output(prediction):   
    formatted_output = []
    current_section = None

    for prediction in predictions:
        section, text = prediction.split(": ", 1)
        if section != current_section:
            if current_section is not None:
                formatted_output.append("\n")
            formatted_output.append(section)
            current_section = section
        formatted_output.append(f" {text}")

    return formatted_output

