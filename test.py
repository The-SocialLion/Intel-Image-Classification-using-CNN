import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image# used for preproccesing 
model = load_model('my_model.h5')
print("Loaded model from disk")

classs = { 1:"buildings",
           2:"forest",
           3:"glacier",
           4:"mountain",
           5:"sea",
           6:"street"}

def classify(img_file):
    test_image=image.load_img(img_file)
    test_image = test_image.resize((30, 30))
    test_image = np.expand_dims(test_image, axis=0)
    test_image = np.array(test_image)

    result = model.predict_classes(test_image)[0]
    sign = classs[result + 1]
    print(sign)
    
print("Obtaining Images & its Labels..............")
path='D:\python\dl programs\intel image classification\data\Test'
files=[]
print("Dataset Loaded")
# r=root,d=directories,f=files
for r,d,f in os.walk(path):
    for file in f:
        if '.jpeg' or '.jpg' or '.png' in file:
            files.append(os.path.join(r,file))
for f in files:
    classify(f)
    print('\n')

