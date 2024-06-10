import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

PAGE_TITLE = "Прогноз риска возникновения лесных пожаров по спутниковому снимку"
PAGE_ICON = ""
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"
MODEL_PATH = "models/CNN_model.h5"
IMAGE_PATH = "data/val_data"
IMAGE_SIZE = (32, 32)
IMAGE_CHANNELS = 3
IMAGE_NORMALIZATION = 255.0

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE,
)

st.header(PAGE_TITLE, divider="orange")

st.caption("Для оценки местности - загрузите фото или выберите из списка")


@st.cache
def load_model_cache() -> tf.keras.Model:
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# new_model: tf.keras.Model = load_model_cache()


def predict_wildfire(image: np.ndarray) -> np.ndarray | None:
    try:
        new_model: tf.keras.Model = load_model(MODEL_PATH)
        img = cv2.resize(image, IMAGE_SIZE)
        img = img / IMAGE_NORMALIZATION
        img = img.reshape(1, *IMAGE_SIZE, IMAGE_CHANNELS)
        predictions = new_model.predict(img)
        return predictions
    except Exception as e:
        print(f"Ошибка: {e}")
        return None


def load_image_from_path(path: str) -> np.ndarray | None:
    try:
        image = cv2.imread(path)
        return image
    except Exception as e:
        print(f"Ошибка: {e}")
        return None


def load_image_from_upload(uploaded_file) -> np.ndarray | None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Ошибка: {e}")
        return None


image_options = [
    f for f in os.listdir("data/val_data") if f.endswith((".jpg", ".jpeg", ".png"))
]

select_or_upload = st.selectbox(
    "Выберите способ загрузки снимка",
    ["Выбрать из списка", "Загрузить свой снимок"],
    index=None,
    placeholder="Select...",
)

if select_or_upload == "Выбрать из списка":
    selected_image = st.selectbox(
        "Выберите снимок для валидации",
        image_options,
        index=None,
        placeholder="Select image...",
    )
    image_path = f"{IMAGE_PATH}/{selected_image}"
    image = load_image_from_path(image_path)
    st.write(f"Выбрано фото: {selected_image}")

elif select_or_upload == "Загрузить свой снимок":
    uploaded_file = st.file_uploader("Загрузите снимок", type=["jpg", "jpeg", "png"])
    image = load_image_from_upload(uploaded_file)

submit_button = st.button("Определить риск возникновения лесных пожаров")

if submit_button:
    try:
        if image is None:
            st.info("Загрузите фото")
        else:
            st.image(
                image,
                caption="Загруженный снимок",
                width=200,
                channels="RGB",
                output_format="auto",
            )
            predictions = predict_wildfire(image)
            st.write(predictions)
            st.write(new_model)
            if predictions is None:
                st.error("Ошибка прогноза")
            else:
                predicted_class = np.argmax(predictions)
                message = (
                    "Район не подвержен риску возникновения лесных пожаров"
                    if predicted_class == 0
                    else "Район подвержен риску возникновения лесных пожаров"
                )
                st.success(message) if predicted_class == 0 else st.warning(message)
    except Exception as e:
        print(f"Ошибка: {e}")
