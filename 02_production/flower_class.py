import argparse
# import fastbook
from fastai.vision.all import *
from fastai.vision.widgets import *
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img')
    args = parser.parse_args()
    img =Image.open(args.img)
    img.show()
    path = Path()
    learn_inf = load_learner(path/'export.pkl', cpu=True)
    pred,pred_idx,probs = learn_inf.predict(args.img)
    print('Prediction:',pred,'| Probability:',"{:.4f}".format(probs[pred_idx]))