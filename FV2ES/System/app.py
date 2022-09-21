import os
import json
import subprocess
import numpy as np
import sys
sys.path.append('../base_iemocap_onetest/main.py')  ## by ling
# Import Flask
from flask import Flask, render_template, request, redirect, jsonify, send_from_directory
from werkzeug.utils import secure_filename  ## Check if the filename is valid

# Create Flask instance
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = '../video'  ## Create upload directory


# The decorator implements route mapping and establishes the association between URL rules and handler functions
# Tell Flask what kind of URL can trigger our function
@app.route('/', methods=['GET'])
def index():
    return render_template('index1.html')

# POST trigger
@app.route('/', methods=['POST', 'GET'])
def upload_function():
    # save file
    if request.method == 'POST':
        # f = request.files['file']
        print(request.files.getlist('file'))
        # save
        for f in request.files.getlist('file'):
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        return ('', 204)
    # todo：返回数据


@app.route('/predict', methods=['POST'])
def pridict_function():
    # call the prediction module and return the prediction result
    cmd = ['python', '../V2EM_prediction/main.py', '--test']
    cmd_result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8')
    if cmd_result.returncode == 0:
        print('success')
        # if successful, read result.txt
        emotion_list = []
        with open('../V2ES_prediction/result.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                emotion_list.append(line)
        print(emotion_list)

        emotion_vector = {'angry': emotion_list[0], 'excited': emotion_list[1], 'frustrated': emotion_list[2],
                      'happy': emotion_list[3], 'neural': emotion_list[4], 'sad': emotion_list[5], }
        data = jsonify(emotion_vector)
        return data, 201, {"ContentType": "application/json"}


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=int(os.environ.get('PORT', 7890)))
