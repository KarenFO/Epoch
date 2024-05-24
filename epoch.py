# Función de activación (función escalón)
def step_function(y):
    return 1 if y >= 0 else 0

# Función para entrenar el perceptrón
def train_perceptron(X, y, learning_rate=0.1, epochs=100):
    # Inicializar los pesos y el bias
    num_features = len(X[0])
    weights = [0.0] * num_features
    bias = 0.0

    for epoch in range(epochs):
        total_error = 0
        for xi, target in zip(X, y):
            # Calcular la salida del perceptrón
            y_in = sum(w * x for w, x in zip(weights, xi)) + bias
            y_pred = step_function(y_in)

            # Calcular el error
            error = target - y_pred
            total_error += abs(error)

            # Actualizar los pesos y el bias si hay error
            if error != 0:
                weights = [w + learning_rate * error * x for w, x in zip(weights, xi)]
                bias += learning_rate * error

        # Imprimir el estado actual de los pesos y el bias
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Weights: {weights}, Bias: {bias}, Total Error: {total_error}')

        # Si el error total es 0, el entrenamiento se detiene
        if total_error == 0:
            break

    return weights, bias

# Definir el conjunto de datos de entrada y las etiquetas
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 0, 0, 1]  # AND gate

# Entrenar el perceptrón
weights, bias = train_perceptron(X, y, learning_rate=0.1, epochs=10)

# Imprimir los pesos y el bias finales
print(f'Final Weights: {weights}, Final Bias: {bias}')

# Calcular la salida final con los pesos entrenados
def predict(X, weights, bias):
    y_pred = []
    for xi in X:
        y_in = sum(w * x for w, x in zip(weights, xi)) + bias
        y_pred.append(step_function(y_in))
    return y_pred

# Probar el perceptrón con los datos de entrada
y_pred = predict(X, weights, bias)
print(f'Predictions: {y_pred}')
