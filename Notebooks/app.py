#---------------------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------------------
import os
import pathlib
import time


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.utils import to_categorical
from yaml import load, Loader, dump




#---------------------------------------------------------------------------------------
# Config page
#---------------------------------------------------------------------------------------
img = Image.open("Notebooks/CTG.png")
st.set_page_config(
     page_title="Test segmentation et suggestion",
    #  page_icon=":shark",
     page_icon=img,
     layout="wide",
     initial_sidebar_state="expanded",
    #  menu_items={
    #      'Get Help': 'https://github.com/MerylAhounou/plane_classification_projet',
    #      'Report a bug': "https://github.com/MerylAhounou/plane_classification_projet",
    #      'About': "# This is an *plane classification* cool app! Let's try!!"
    #  }
 )



#---------------------------------------------------------------------------------------
# Head app
#---------------------------------------------------------------------------------------
# st.title("Identifation d'avion")
html_temp = """
<div style="background-color:blue;padding:1.5px">
<h1 style="color:white;text-align:center;">Segmentation client et suggestion de produit </h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)









#---------------------------------------------------------------------------------------
# Loading files
#---------------------------------------------------------------------------------------
yaml_file = open("app.yaml", 'r')
yaml_content = load(yaml_file, Loader=Loader)



#---------------------------------------------------------------------------------------
# Constantes
#---------------------------------------------------------------------------------------
MODELS_DIR = yaml_content["MODELS_DIR"]


PATH_CLASSES = pathlib.Path(MODELS_DIR +'\\categories.txt')


classes_names_list = []
with open(PATH_CLASSES, 'r') as f:
    for name in f.readlines():
        classes_names_list.append(name.replace('\n', ''))
#---------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------
@st.cache(ttl = 12*3600, allow_output_mutation=True, max_entries=5)
def load_model(path, type_model):
    """Load tf/Keras model for prediction
    Parameters
    ----------
    path (str): Path to model
    type_model (int): type of model to use for prediction
                    0 RNN
                    1 Others models
    Returns
    -------
    Predicted class
    """
    if type_model ==0:
      return tf.keras.models.load_model(path)
    elif type_model ==1:
      return pickle.load(open(path,'rb'))
  
  
  
  

def vect_cluster(seq, path):
    """Allows to encode sequences with tf-idf, pca and to perform clustering

    Args:
        seq (seq): clustering sequence
        path (str): Path to directory of different models
    """
    path_tfidf = path + "/" + "tfidf.p"
    path_pca = path + "/" + "pca.p"
    path_kmeans = path + "/" + "kmeans.p"
    # path_tfidf = "tfidf.p"
    # path_pca =  "pca.p"
    # path_kmeans =  "kmeans.p"
    model_tfidf = load_model(path_tfidf, 1)
    model_pca = load_model(path_pca, 1)
    model_kmeans = load_model(path_kmeans, 1)
    vec = model_tfidf.transform([seq])
    pca = model_pca.transform(vec.toarray())
    kmeans = model_kmeans.predict(pca)
    cluster = kmeans[0]
    return cluster




def rnn(seq, path, path_classes):
    """Allows to perform the rnn model prediction

    Args:
        seq (seq): rnn sequence
    """
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    path_rnn = path + "/" + "rnn.h5"
    # path_rnn =  "rnn.h5"
    model_rnn = load_model(path_rnn, 0)
    names = pd.read_csv(path_classes, names=['Names'])
    df = pd.DataFrame([seq])
    X = to_categorical(df[0].astype('category').cat.codes)
    # st.dataframe(df[0])
    tensor = np.array([np.array(x) for x in X])
    prediction_vector = model_rnn.predict(tensor)
    predicted_classes = np.argmax(prediction_vector, axis=1)[0]
    predicted_prob = prediction_vector[0][predicted_classes] * 100
    name_classes = names['Names'][predicted_classes]
    # return predicted_classes, predicted_prob, prediction_vector
    return predicted_classes, predicted_prob, prediction_vector, name_classes
    
    
    
    
st.info(f'Liste des différentes actes: {classes_names_list}')
#---------------------------------------------------------------------------------------
# Columns
#---------------------------------------------------------------------------------------
col1, col2 = st.columns(2)


#---------------------------------------------------------------------------------------
# Sequence insertion clustering
#---------------------------------------------------------------------------------------
with col1:
    html_temp = """
    <h2 style="color:white;text-align:left;">Segmentation client</h2>
    </div><br>"""
    st.markdown(html_temp,unsafe_allow_html=True)



    st.info('NB: Une séquence pour le clustering doit être constitué de 5 actes')
    seq_cluster = st.text_input('Séquence pour clustering', )
    st.write('La séquence est', seq_cluster)
    
    
    clust_btn = st.button('Clustering', disabled=(seq_cluster ==""))

    
    


#---------------------------------------------------------------------------------------
# Sequence insertion RNN
#---------------------------------------------------------------------------------------
with col2:
    html_temp = """
    <h2 style="color:white;text-align::left;">Suggestion de produit</h2>
    </div><br>"""
    st.markdown(html_temp,unsafe_allow_html=True)


    st.info('NB: Une séquence pour le modèle de suggestion doit être constitué de 4 actes')
    seq = st.text_input('Séquence pour modèle RNN', )
    st.write('La séquence est', seq)

    predict_btn = st.button('Prédiction', disabled=(seq ==""))
    prob_btn = st.button('Afficher les probabilités', disabled=(seq ==""))



#---------------------------------------------------------------------------------------
# Prediction
#---------------------------------------------------------------------------------------
with col1:
    if clust_btn:
        st.write("Cette individu est dans le cluster", vect_cluster(seq_cluster, MODELS_DIR))
        st.balloons()


with col2:
    if predict_btn:
        my_bar = st.progress(0)
        for percent_complete in range(100):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1)
        prediction_classes, predicted_prob, prediction_vector,prediction_names = rnn(seq, MODELS_DIR, PATH_CLASSES)
        st.write(f"C'est individu serait intéressé par cette prestation: {prediction_names} avec une\
                probabilité de prédiction de:{predicted_prob: .2f}%")
        st.balloons()
      

  
with col2:
    if prob_btn:
        my_bar = st.progress(0)
        for percent_complete in range(100):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1)
        prediction_classes, predicted_prob, prediction_vector,prediction_names = rnn(seq, MODELS_DIR, PATH_CLASSES)
        prediction_vector = prediction_vector*100
        chart_data = pd.DataFrame(prediction_vector, columns= classes_names_list).T
        st.bar_chart(chart_data)
        st.balloons()
    

#---------------------------------------------------------------------------------------
# Tail
#---------------------------------------------------------------------------------------
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<b><div align="center">**``AHOUNOU Méryl``**</div></b>',
            unsafe_allow_html=True)
