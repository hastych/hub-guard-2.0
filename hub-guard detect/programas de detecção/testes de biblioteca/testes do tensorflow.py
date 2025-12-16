import tensorflow as tf
from tensorflow.keras import layers, models

# Carrega o dataset MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normaliza os dados
x_train, x_test = x_train / 255.0, x_test / 255.0

# Cria o modelo
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10)
])

# Compila o modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Treina o modelo
model.fit(x_train, y_train, epochs=5)

# Avalia o modelo
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nAcurácia no teste: {test_acc}')