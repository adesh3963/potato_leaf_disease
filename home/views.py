from django.shortcuts import render

import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img
import numpy as np
from .models import picturs


class_name=['Early Blight Leaf', 'Late Blight Leaf', 'Healthy Leaf']
main_model= load_model("./final_model.h5")
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array= tf.expand_dims(img_array, 0) #creating a batch
    prediction = main_model.predict(img_array)
    
    predicted_class= class_name[np.argmax(prediction[0])]
    confidence = round(100* (np.max(prediction[0])),2)
    return predicted_class, confidence
    

# Create your views here.
def base(request):
    if request.method == 'POST':
        print("file coming")
        print(request.FILES)
        img = request.FILES['data'] 
        obj= picturs(pic= img)
        obj.save()
        print("Image saved")

        pics= picturs.objects.all()
        p= pics[len(pics)-1].pic

        #image_name= img.name

        #all_files= os.listdir("./media/images")
        #temp_path= "\media\images"+ image_name
        #image_path= os.path.join(BASE_DIR,temp_path)
        
        #converted_path= Path(p.url)
        #print(BASE_DIR)
        #image_path=BASE_DIR + converted_path
        #print("chaek 2")
        
        #print(image_path)
        image_path= "."+p.url
        
        image= load_img(image_path)
        
        result,confi= predict(main_model,image) 
        context={"result":result,"confidence":confi,"pic":p.url}
        return render(request,"base.html",context)
    else:
        return render(request,"base.html")
    