from flask import *
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Classes of trafic signs
classes = { 
            0:'Doctor',
            1:'Hands', 
            2:'Itch', 
            3:'Maximum', 
            4:'Wednesday', 
            5:'Welcome', 
            6:'Wood' 
 }

def image_processing(img):
    model = load_model('./model/TSR.h5')
    data=[]
    image = Image.open(img)
    image = image.resize((50,50))
    data.append(np.array(image))
    X_test=np.array(data)
    pred = model.predict(X_test)
    Y_pred = np.argmax(pred,axis=1)
    return Y_pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ISL-Symbols',methods=['GET'])
def isl():
    return render_template('ISL-Symbols.html')

@app.route('/profiles')
def prof():
    return render_template('profiles.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = image_processing(file_path)
        s = [str(i) for i in result]
        a = int("".join(s))
        result = "Predicted sign is: " +classes[a]
        os.remove(file_path)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)