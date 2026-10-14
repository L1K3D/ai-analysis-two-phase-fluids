from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_fast_cnn():
    model = Sequential([
        # Camada de convolução para detetar os contornos das bolhas de ar/vapor
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        
        # Segunda extração de características (turbulência do fluido)
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Achatamento do tensor para a camada de decisão
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5), # Previne o overfitting (que o modelo apenas decore as imagens)
        Dense(1, activation='sigmoid') # Saída binária: 0 (Expulsion) ou 1 (Incubation)
    ])
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model