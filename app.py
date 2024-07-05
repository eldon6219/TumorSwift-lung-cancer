import tensorflow as tf
import numpy as np
from flask import Flask , render_template , request
from keras.optimizers import Adam, Adamax
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
model = load_model("models/lung.h5")
model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

class_labels = ['Colon Adenocarcinoma','Colon Benign Tissue','Lung Adenocarcinoma','Lung Benign Tissue','Lung Squamous Cell Carcinoma']




app = Flask(__name__) 

@app.route('/', methods=['GET'])
def hello_world():
return render_template("index.html")

 @app.route('/predict' , methods=['POST'])
 def predict():

# Get the values from the form in the request
imagefile=request.files['imagefile']
image_path = "./img/" + imagefile.filename
imagefile.save(image_path)

 image = load_img(image_path , target_size=(224,224,3))
 image = img_to_array(image)
 image = np.expand_dims(image, axis=0)


 prediction = model.predict(image)
 score = tf.nn.softmax(prediction[0])
 predicted_class_index = np.argmax(score)
 predicted_class_label = class_labels[predicted_class_index]


 return render_template("result.html" , predictions=predicted_class_label)

if __name__ == '__main__':
    app.run(debug=False )




