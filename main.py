import os
import pickle
import numpy as np
import tensorflow
from keras.api.preprocessing import image
from keras.src.layers import GlobalMaxPooling2D
from keras.src.applications.resnet import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from PIL import Image
from numpy.linalg import norm
st.title('Fashion Recommender system')


def save_uploaded_file(uploaded_file):
    try:
        # Ensure the uploads directory exists
        os.makedirs('uploads', exist_ok=True)
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False


feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))

filenames = pickle.load(open('filenames.pkl','rb'))

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


def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])
    return indices


uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        feature = extract_features(os.path.join("uploads",uploaded_file.name),model)
        indices= recommend(feature, feature_list)

        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][1]])
        with col2:
            st.image(filenames[indices[0][2]])
        with col3:
            st.image(filenames[indices[0][3]])
        with col4:
            st.image(filenames[indices[0][4]])
        with col5:
            st.image(filenames[indices[0][5]])


    else:
        st.error("Some error occurred while saving the file.")


