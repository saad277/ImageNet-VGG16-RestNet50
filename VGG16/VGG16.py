
from keras.preprocessing import image
from keras.applications import vgg16
from keras.applications.vgg16 import decode_predictions,preprocess_input
import numpy as np



vgg16_model=vgg16.VGG16(weights="imagenet");


print("working");


img_path="./images/dog.jpg"

img=image.load_img(img_path,target_size=(224,224))

x=image.img_to_array(img)

x=np.expand_dims(x,axis=0);

x=preprocess_input(x)

preds=vgg16_model.predict(x)

print(preds)

print("Predicted :",decode_predictions(preds,top=3)[0])
