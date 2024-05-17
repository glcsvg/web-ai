from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import io
import werkzeug
import datetime
import os

model = ResNet50(weights="imagenet")

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST" and request.files['image']:
        imagefile = request.files["image"].read()
        image = Image.open(io.BytesIO(imagefile))
        # preprocess the image and prepare it for classification
        image = prepare_image(image, target=(224, 224))
        # classify the input image and then initialize the list
        # of predictions to return to the client
        preds = model.predict(image)
        results = imagenet_utils.decode_predictions(preds)
        data["predictions"] = []
        # loop over the results and add them to the list of
        # returned predictions
        for (imagenetID, label, prob) in results[0]:
            r = {"label": label, "probability": float(prob)}
            data["predictions"].append(r)
        # indicate that the request was a success
        data["success"] = True
        print(data)
# return the data dictionary as a JSON response
    return jsonify(data)

app = Flask(__name__)

@app.route("/")
def main():
    return render_template ("index.html")
if __name__ == "__main__":
    print("START FLASK")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    #app.run(debug=True)