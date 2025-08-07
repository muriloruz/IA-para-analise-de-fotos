import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (MobileNetV2, preprocess_input, decode_predictions as d)
# TensorFlow com Keras possui embutido muitos modelos prontos de ML
from PIL import Image

# Essa função serve para carregar o modelo dentro desse MobileNetV2 
def load_model():
    model = MobileNetV2(weights='imagenet') # Carrega o modelo MobileNetV2 com pesos pré-treinados no ImageNet
    return model

# Função para pré-processar a imagem antes de classificá-la
def preprocess_image(image):
    img = np.array(image)  # Converte a imagem PIL para um array NumPy
    img = cv2.resize(img, (224, 224))  # Redimensiona a imagem para 224x224 pixels
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0) # Transforma a imagem para uma lista de imagem
    return img

# Função para classificar a imagem usando o modelo MobileNetV2
def classify_image(model, image):
    try:
        preprocessed_image = preprocess_image(image)
        predictions = model.predict(preprocessed_image)  # Faz a predição
        decode_predictions = d(predictions, top=1)[0]  # Decodifica as predições com base na única imagem enviada
        return decode_predictions
    except Exception as e:
        st.error(f"Erro ao classificar a imagem: {str(e)}")
        return None
    
def main():
    st.set_page_config(page_title="Classificador de Imagens", page_icon=":📸:", layout="centered")
    st.title("Classificador de Imagens com IA")
    st.write("Carregue uma imagem e veja como a IA a classifica!")

    @st.cache_resource # Serve para não ficar carregando o modelo toda hora
    def load_cached_model():
        return load_model()
    
    model = load_cached_model()

    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = st.image(uploaded_file, caption='Imagem Carregada.', use_column_width=True)
        btn = st.button("Classificar Imagem")
        if btn:
            with st.spinner('Classificando...'):
                image = Image.open(uploaded_file)
                predictions = classify_image(model,image)
                if predictions:
                    st.subheader("Classificação Completa!")
                    for _, label, score in predictions:
                        st.write(f"Com base na análise, fica claro que isso é um(a) \"{label}\".\nA porcentagem de certeza é de: {score*100:.2f}%")


if __name__ == "__main__":
    main()