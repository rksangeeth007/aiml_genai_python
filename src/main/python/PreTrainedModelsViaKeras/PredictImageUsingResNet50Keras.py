from IPython.display import Image
from IPython import display

Image(
    filename='/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy_repo/src/main/resources/ImagesForPredictionCNN/fighterjet.jpg')

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

img_path = '/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy_repo/src/main/resources/ImagesForPredictionCNN/fighterjet.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

model = ResNet50(weights='imagenet')
preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])


def classify(img_path):
#    display(Image(filename=img_path))

    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=3)[0])


classify(
    '/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy_repo/src/main/resources/ImagesForPredictionCNN/bunny.jpg')
classify(
    '/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy_repo/src/main/resources/ImagesForPredictionCNN/bunny.jpg')
classify(
    '/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy_repo/src/main/resources/ImagesForPredictionCNN/firetruck.jpg')
classify(
    '/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy_repo/src/main/resources/ImagesForPredictionCNN/breakfast.jpg')
classify(
    '/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy_repo/src/main/resources/ImagesForPredictionCNN/castle.jpg')
classify(
    '/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy_repo/src/main/resources/ImagesForPredictionCNN/VLA.jpg')
classify(
    '/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy_repo/src/main/resources/ImagesForPredictionCNN/bridge.jpg')


# classify(
#     '/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy_repo/src/main/resources/PersImagesForPredictionCNN/20230527_102909.jpg')
# classify(
#     '/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy_repo/src/main/resources/PersImagesForPredictionCNN/20230527_133931.jpg')
# classify(
#     '/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy_repo/src/main/resources/PersImagesForPredictionCNN/20230527_140337.jpg')
# classify(
#     '/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy_repo/src/main/resources/PersImagesForPredictionCNN/20230527_165617.jpg')
# classify(
#     '/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy_repo/src/main/resources/PersImagesForPredictionCNN/20230527_165623.jpg')
# classify(
#     '/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy_repo/src/main/resources/PersImagesForPredictionCNN/20230527_172805.jpg')
# classify(
#     '/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy_repo/src/main/resources/PersImagesForPredictionCNN/20230527_173421.jpg')