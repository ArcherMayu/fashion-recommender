import numpy as np
import tensorflow
from keras.src.layers import GlobalMaxPooling2D
from tensorflow import keras
import pickle
import numpy as np
from numpy.linalg import norm
from keras.api.preprocessing import image
from keras.src.applications.resnet import ResNet50,preprocess_input
import os
from tqdm import tqdm


model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
model.trainable= False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_path,model):
   img = image.load_img(img_path,target_size=(224,224))
   img_array = image.img_to_array(img)
   expanded_img_array = np.expand_dims(img_array,axis=0)
   prepped_input = preprocess_input(expanded_img_array)

   result = model.predict(prepped_input).flatten()
   normalized_result = result/norm(result)

   return normalized_result



filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))



feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))


