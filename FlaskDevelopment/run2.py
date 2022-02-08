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
btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()


app = Flask(__name__)

@app.route('/success/<name>')
def success(pic):
    lbl_pred.value = ''
    img = PILImage.create(pic[-1])
    out_pl.clear_output()
    # with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

    return '%s' % lbl_pred.value

@app.route('/login',methods = ['POST', 'GET'])
def login():
    if request.method == 'POST':
        # print(request.files)
        # # print(request.files['imgInp'])
        # picture = request.files.get('imgInp')
        # img = Image.open(picture)
        # display(img.to_thumb(128,128))

        filestr = request.files['imgInp'].read()
        npimg = np.fromstring(filestr, np.uint8)
        # print(npimg)
        picture = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        picture = cv2.cvtColor(picture, cv2.COLOR_RGB2BGR)
        # print(picture)
        img = Image.fromarray(picture, 'RGB')
        img.show()
        # # img = PILImage.create(picture[-1])
        # print('\n\nllego aqui 2',img,'\n\n')
        # # out_pl.clear_output()
        # print('Picture and img variables:',picture,img)
        # with out_pl: display(img.to_thumb(128,128))
        # return redirect(url_for('success',name = picture))
    else:
        user = request.args.get('nm')
        return redirect(url_for('success',name = user))

if __name__ == '__main__':
    app.run(debug = True)






