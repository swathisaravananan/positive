from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64
import uuid
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from keras.models import load_model
import numpy as np

module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
print("movenet imported")

app = Flask(__name__)
cors = CORS(app, resources={r"/foo": {"origins": "*"}}, support_credentials = True)
app.config['CORS_HEADERS'] = 'Content-Type'

class_model = load_model('nn_saved.h5')

class_labels = ["side-bend", "squatting", "standing"]

def get_keypoints(image_path):
    model = module.signatures['serving_default']
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    input_image = tf.expand_dims(image, axis=0)
    input_size = 256
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = model(input_image)
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

def get_class(image_path):
    kp = get_keypoints(image_path)
    keypoints_with_scores = get_keypoints(image_path)
    reshaped = tf.reshape(keypoints_with_scores, (1,51))
    aug = reshaped.numpy().tolist()[0]
    new_aug = []
    for i in range(len(aug)):
        if (i+1) % 3 == 0:
            new_aug.append(aug[i])
        else:
            new_aug.append(aug[i]*1000)
    new_t = tf.convert_to_tensor(new_aug)
    reshaped = tf.reshape(new_t, (1,51))
    p1 = class_model.predict(reshaped)
    print(reshaped)
    p_1 = [np.argmax(px) for px in p1]
    return class_labels[p_1[0]]


@app.route("/image", methods=["POST"])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def classify_image():
    try:
        filename =  '/Users/sriraja/Desktop/GAIP/project/chatbot-server/requests/'+str(uuid.uuid4())+'.jpg'
        recvd_file = request.files['file']
        recvd_file.save(filename)
        class_name = get_class(filename)
        print(class_name)
        return jsonify({
            "status": "ok"
        }) 

    except:
        print("error occured")
        return jsonify({
            "status": "error"
        }) 



if __name__ == "__main__":
    app.run()


# @app.route("/query", methods=["POST"])
# @cross_origin(origin='*',headers=['Content-Type','Authorization'])
# def query():
#     question = request.json["question"]
#     print(question)
#     return jsonify({
#         "type" : "recvd",
#         "headline" : "This is the headline",
#         "content" : "These are `suggestions on how to correct the posture."
#     })
