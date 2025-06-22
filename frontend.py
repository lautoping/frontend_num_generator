import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar el modelo entrenado (Keras)
model = tf.keras.models.load_model("tensorflow_mnist.h5")

st.set_page_config(page_title="MNIST Classifier", layout="centered")
st.title("🖊️ Clasificador de Dígitos Manuscritos MNIST")
st.write("Subí una imagen de un dígito manuscrito (28x28 píxeles, escala de grises) o probá con imágenes reales del dataset.")

option = st.radio("¿Qué querés hacer?", ["Subir imagen", "Probar con imágenes reales del dataset"])

if option == "Subir imagen":
    uploaded_file = st.file_uploader("Elegí una imagen PNG o JPG (28x28, escala de grises)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("L").resize((28, 28))
        img_arr = np.array(img).astype("float32") / 255.0
        img_arr = img_arr[None, ..., None]  # (1, 28, 28, 1)
        pred = model.predict(img_arr)
        pred_label = np.argmax(pred, axis=1)[0]
        st.image(img, caption=f"Predicción: {pred_label}", width=128)
        st.write(f"El modelo predice: **{pred_label}**")

else:
    # Probar con imágenes reales del dataset MNIST
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype('float32') / 255.0
    x_test = x_test[..., None]
    digit = st.selectbox("Elegí un dígito (0-9):", list(range(10)))
    num_samples = st.slider("Cantidad de muestras", 1, 10, 5)
    indices = np.where(y_test == digit)[0][:num_samples]
    imgs = x_test[indices]
    preds = model.predict(imgs)
    pred_labels = np.argmax(preds, axis=1)
    cols = st.columns(num_samples)
    for i, col in enumerate(cols):
        img = (imgs[i].squeeze() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img, mode='L').resize((128, 128), Image.NEAREST)
        col.image(img_pil, caption=f"Pred: {pred_labels[i]}", use_column_width=True)