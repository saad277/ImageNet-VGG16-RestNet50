
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications import vgg16
import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile,join



vgg16_model=vgg16.VGG16(weights="imagenet");


print("working");



def draw_test(name,prediction,input_image):
    BLACK=[0,0,0];
    expanded_image=cv2.copyMakeBorder(input_image,0,0,0,imageL.shape[1]+700,cv2.BORDER_CONSTANT,value=BLACK)
    img_width=input_image.shape[1];

    for (i,prediction) in enumerate(predictions):
        string=str(prediction[1]+" "+str(prediction[2]));
        cv2.putText(expanded_image,str(name),(img_width+50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255),1)
        cv2.putText(expanded_image,string,(img_width+50,50+((i+1)*50)),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),0)
        cv2.imshow(name,expanded_image)


mypath="./images/"

file_names=[f for f in listdir(mypath) if isfile(join(mypath,f))]

print(file_names)


for file in file_names:


    img=image.load_img(mypath+file,target_size=(224,224))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)



    img2=cv2.imread(mypath+file);
    imageL=cv2.resize(img2,None,fx=.5,fy=.5,interpolation=cv2.INTER_CUBIC)

    preds=vgg16_model.predict(x);
    predictions=decode_predictions(preds,top=3)[0]
    print(predictions)
    draw_test("Predictions",predictions,imageL);
    cv2.waitKey(0);


cv2.destroyAllWindows();
















