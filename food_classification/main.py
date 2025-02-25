import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


# Sidebar
st.sidebar.title("dashboard")
app_mode = st.sidebar.selectbox("select the page", ["home", "about project", "prediction"])

# Main page
if(app_mode == "home"):
    st.header("fruits and vegetable prediction")
    image_path = "home_img.jpg"
    st.image(image_path)
elif (app_mode == "about project"):
    st.header("about project")
    st.subheader("about dataset")
    st.text("""
                from: 'fruits and vegetables image recognition dataset' (https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)
                
                this dataset encompasses images of various fruits and vegetables, providing a diverse collection for image recognition tasks. the included food items are:
            """)
    st.code("fruits: banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango")
    st.code("vegetables: cucumber, carrot, capsicum, onion, potato, lemon, tomato, radish, beetroot, cabbage, lettuce, spinach, soybean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet Potato, paprika, jalapeño, ginger, garlic, peas, eggplant")
    st.subheader("content of the dataset:")
    st.code("train: contains 100 images per category.")
    st.code("test: contains 10 images per category.")
    st.code("validation: contains 10 images per category.")
    
elif(app_mode=="prediction"):
    st.header("model prediction")
    test_image = st.file_uploader("choose an image:")
    if(st.button("show image")):
        st.image(test_image,width=4,use_container_width=True)
    #Predict button
    if(st.button("predict")):
        st.snow()
        st.write("our prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        with open("labels.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        st.success(f"it's a(n) {label[result_index]}")