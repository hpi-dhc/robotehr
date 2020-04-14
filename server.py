import os

from flask import Flask, jsonify, request
import pandas as pd

from robotehr.storage.load import load_model

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
     json_ = request.json
     query_df = list(json_.values())
     # TODO: Transform data!
     prediction = clf.predict([query_df])
     return jsonify({'prediction': list(prediction)})


if __name__ == '__main__':
     clf = load_model(os.environ.get('ROBOTEHR_SERVER_MODEL_ID'))
     app.run(port=8080, debug=True)
