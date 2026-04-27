import streamlit as st

import numpy as np

from PIL import Image

import tflite_runtime.interpreter as tflite # يتطلب بايثون 3.10 أو 3.11



st.set_page_config(page_title="نظام تصنيف الزهور الذكي", page_icon="🌸")

st.title("🌸 نظام تصنيف الزهور الذكي")



def predict_flower(image_data):

    # تحميل المفسر الحقيقي لملف tflite

    interpreter = tflite.Interpreter(model_path="model.tflite")

    interpreter.allocate_tensors()

    

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    

    # معالجة الصورة

    img = image_data.resize((128, 128))

    img_array = np.array(img, dtype=np.float32) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    

    # التوقع الحقيقي

    interpreter.set_tensor(input_details[0]['index'], img_array)

    interpreter.invoke()

    return interpreter.get_tensor(output_details[0]['index'])[0]



class_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

uploaded_file = st.file_uploader("اختر صورة...", type=["jpg", "png", "jpeg"])



if uploaded_file:

    image = Image.open(uploaded_file).convert('RGB')

    st.image(image, use_container_width=True)

    

    predictions = predict_flower(image)

    result = class_names[np.argmax(predictions)]

    

    st.success(f"✅ النتيجة الحقيقية من النموذج: {result}")