from fastapi import FastAPI,File,UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

model = tf.keras.models.load_model("plantvillage.h5")
class_names = ['Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites Two spotted spider mite', 'Tomato Target Spot', 'Tomato YellowLeaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

def convert2image(data):
    numpy_image = np.array(Image.open(BytesIO(data)))
    return numpy_image

app = FastAPI()

@app.get('/hello')
async def hello():
    return "Hello!! I am still Working!!!!!!!!!!"

@app.post('/predict_plant')
async def predict_plant(
    file: UploadFile = File(...)
):
    bytes = await file.read()
    image = convert2image(bytes)
    image = np.expand_dims(image,0)
    predict = model.predict(image)
    prediction_class = class_names[np.argmax(predict[0])]
    confidence = np.max(predict[0])
    return {
        'Class':prediction_class,
        'Confidence of Prediction':float(confidence)
    }


if __name__ == '__main__':
    uvicorn.run(app,host='localhost',port=8000)