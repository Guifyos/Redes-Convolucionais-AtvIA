import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image

# Criando a estrutura da Rede Convolucional (CNN) exigida
def criar_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax') # 3 classes: Moto, Bicicleta, Carro
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

cnn_model = criar_cnn()
classes = ["Moto", "Bicicleta", "Carro"]

def processar_imagem(caminho_imagem):
    img = Image.open(caminho_imagem).convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def classificar_imagem(caminho_imagem):
    # Processa a imagem de teste
    img_array = processar_imagem(caminho_imagem)
    
    # Faz a predição usando a CNN
    predicoes = cnn_model.predict(img_array)
    indice_classe = np.argmax(predicoes[0])
    
    # Retorna qual dos 3 é
    return classes[indice_classe]

def treinar_modelo_simples():
    # Esta função existiria para treinar as 3 imagens iniciais
    pass