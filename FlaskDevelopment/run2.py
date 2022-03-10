from flask import Flask, redirect, url_for, request
from fastai.vision.all import *
from fastai.vision.widgets import *
from IPython import display
from pathlib import Path,PosixPath
from platform import system
import numpy as np
import cv2
from PIL import Image

plt = system()
if plt == 'Linux': WindowsPath = PosixPath

path = Path()
learn_inf = load_learner(path/'export.pkl', cpu=True)
# btn_upload = widgets.FileUpload()
# out_pl = widgets.Output()
lbl_pred = widgets.Label()

app = Flask(__name__)

@app.route('/success/<pic>')
def success(pic):
    print(type(pic))
    lbl_pred.value = ''
    img = pic
    # out_pl.clear_output()
    print('\n\n\n llego aqui 2',img,'\n\n\n' )
    # with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    print('\n\n\n llego aqui 3\n\n\n' )
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
    print(lbl_pred.value)
    return '%s' % lbl_pred.value

@app.route('/login',methods = ['POST', 'GET'])
def login():
    if request.method == 'POST':

        filestr = request.files['imgInp'].read()
        npimg = np.fromstring(filestr, np.uint8)
        picture = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        # picture = cv2.cvtColor(picture, cv2.COLOR_RGB2BGR)

        img = PILImage.create(picture)
        pred,pred_idx,probs = learn_inf.predict(img)
        lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
        print(lbl_pred.value,probs)

        return '%s' % lbl_pred.value

        # return redirect(url_for('success',pic = img))
    else:
        print('here')
        user = request.args.get('nm')
        return redirect(url_for('success',pic = user))

if __name__ == '__main__':
    app.run(debug = True)






