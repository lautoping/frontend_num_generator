import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar el modelo entrenado (Keras)
model = tf.keras.models.load_model("tensorflow_mnist.h5")

st.set_page_config(page_title="MNIST Classifier", layout="centered")
st.title("🖊️ Clasificador de Dígitos Manuscritos MNIST")
st.write("Dibujá o subí una imagen de un dígito manuscrito (1-9, 28x28 píxeles, escala de grises).")

uploaded_file = st.file_uploader("Elegí una imagen PNG o JPG (28x28, escala de grises)", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L").resize((28, 28))
    img_arr = np.array(img).astype("float32") / 255.0
    img_arr = img_arr[None, ..., None]  # (1, 28, 28, 1)
    pred = model.predict(img_arr)
    pred_label = np.argmax(pred, axis=1)[0]
    if 1 <= pred_label <= 9:
        st.image(img, caption=f"Predicción: {pred_label}", width=128)
        st.write(f"El modelo predice: **{pred_label}**")
    else:
        st.image(img, caption="Predicción fuera de rango (solo 1-9)", width=128)
        st.warning("El modelo solo acepta dígitos del 1 al 9. Intenta con otra imagen.")
else:
    st.info("Subí una imagen de un dígito manuscrito entre 1 y 9.")
