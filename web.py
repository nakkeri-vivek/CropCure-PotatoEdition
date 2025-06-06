import streamlit as st
import numpy as np 
import tensorflow as tp

def model_prediction(test_image) :
    model = tp.keras.models.load_model('trained_plant_disease_model.keras')
    image = tp.keras.preprocessing.image.load_img(test_image, target_size= (128,128))
    input_arr = tp.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    predicted_index = np.argmax(predictions)  
    confidence = predictions[0][predicted_index]
    return predicted_index, confidence
st.sidebar.title("Plant disease System for sustainable Agriculture : ")
app_mode  = st.sidebar.selectbox('select page', ['Home', 'Disease Recognition'])

from PIL import Image
img = Image.open('photo.png')
st.image(img)

if app_mode == "HOME" :
    st.markdown("<h1 style  = 'text-align :center ;'>Plant Disease Detection System for Sustainable Agriculture", unsafe_allow_html=True)
elif app_mode == "Disease Recognition" :
    st.header("Plant Disease Detection System for Sustainable Agriculture ")
test_image = st.file_uploader("Choose an image : ")
if(st.button ('Show image')) :
    st.image(test_image, width=4, use_column_width=True)

if (st.button("Predict")) :
    st.snow()
    st.write('our prediction')
    result_index, confidence =(model_prediction(test_image))

    class_name = ['Potato___Early_blight','Potato___healthy','Potato___Late_blight' ]
    st.success('Model is predicting its a {}'.format(class_name[result_index]))
    st.info(f"Confidence Score: {confidence:.2%}")


