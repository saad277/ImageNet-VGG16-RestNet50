from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np



resnet_model=ResNet50(weights="imagenet");


print("working");


img_path="./images/cat.jpg"

img=image.load_img(img_path,target_size=(224,224))

x=image.img_to_array(img)

x=np.expand_dims(x,axis=0);

x=preprocess_input(x)

preds=resnet_model.predict(x)

#print(preds)

print("Predicted :",decode_predictions(preds,top=3)[0])
