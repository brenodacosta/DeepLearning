#!/usr/bin/env python
# coding: utf-8

# In[7]:


from fastai.vision.all import *
from fastai.vision.widgets import *


# In[8]:


import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

path = Path()
learn_inf = load_learner(path/'export.pkl', cpu=True)
btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()


# In[9]:


def on_data_change(change):
    lbl_pred.value = ''
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'


# ## Flower classifier
# 
# Using fast.ai library, this little application can recognize 5 different types of flowers: daisies, dandelions, roses, sunflowers, and tulips. The way it works cannot be simpler: just upload an flower picture and it will return which flower is it with a probability to be right. As it is just a initial application, if you upload an image which is not a flower (or a non-listed kind of flower), it will give a wrong answer, but showing a small probability of this to be right.

# In[10]:


btn_upload.observe(on_data_change, names=['data'])


# In[11]:


display(VBox([widgets.Label('Select your flower'), btn_upload, out_pl, lbl_pred]))


# In[ ]:




