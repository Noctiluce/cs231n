import numpy as np
import matplotlib.pyplot as plt
import os


#____________________________________
# DEFINES
POINT_PER_CLASS = 100
DIMENSION = 2 # better to not touch this one
NUMBER_OF_CLASSES = 3
SPIRAL_SHAPE = True
SHOW_PROGRESS_AREAS = True
SHOW_PROGRESS_GRAPHIS = False
HIDDEN_LAYER_SIZE = 100
EPOCHS = 3000
LEARNING_RATE = 0.5
HIDDEN_LAYER_COUNT = 0 # there is 1 layer by default even when this is 0
#____________________________________

class Layer:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

end_plot_points = []

#____________________________________
# FUNCTIONS
# randomly generate points in space
def generate_data():
    np.random.seed(0)
    N = POINT_PER_CLASS
    D = DIMENSION
    K = NUMBER_OF_CLASSES
    positions = np.zeros((N*K, D))
    labels = np.zeros(N*K, dtype='uint8')
    for j in range(K):
        ix = range(N*j, N*(j+1))
        if SPIRAL_SHAPE:
            r = np.linspace(0.0, 2, N)  # radius
            t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2  # theta
            positions[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        else:
            positions[ix] = np.random.randn(N, D) + np.array([j * 2, j * 2])  # décale chaque classe
        labels[ix] = j
    return positions, labels

# return z if above 0, 0 otherwise
def relu(z):
    return np.maximum(0, z)

# Softmax
def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # stabilité numérique
    return e_z / np.sum(e_z, axis=1, keepdims=True)

# Forward pass
def forward(X, i_layers):

    o_activations = []
    previous_activation = 0
    lastZ = None
    for i in range(len(i_layers)):

        if i == 0 :
            lastZ = X @ i_layers[i].weight + i_layers[i].bias
        else:
            lastZ = previous_activation @ i_layers[i].weight + i_layers[i].bias

        a = relu(lastZ)
        previous_activation = a
        o_activations.append(a)


    probs = softmax(lastZ)

    return o_activations, probs

# Backward pass
def backward(X, labels, activations, probs, i_layers):
    num_examples = labels.shape[0]
    num_layers = len(i_layers)

    gradients = []

    delta = probs.copy()
    delta[range(num_examples), labels] -= 1
    delta /= num_examples

    for i in reversed(range(num_layers)):
        a_prev = X if i == 0 else activations[i-1]
        W = i_layers[i].weight


        dW = a_prev.T @ delta
        db = np.sum(delta, axis=0, keepdims=True)
        gradients.insert(0, (dW, db))

        if i > 0:
            delta = delta @ W.T
            delta[activations[i-1] <= 0] = 0  # ReLU derivative

    return gradients

# Training model
def train(i_positions, i_labels, i_layers, epochs=1000, learning_rate=1.0):

    num_examples = POINT_PER_CLASS*NUMBER_OF_CLASSES
    input_size = DIMENSION

    plotAreas(positions, labels, i_layers,0)

    activations, probs = forward(i_positions, i_layers)
    loss = -np.sum(np.log(probs[range(num_examples), i_labels])) / num_examples
    print(f"Epoch {0}, loss: {loss:.4f}")
    end_plot_points.append((0, loss))

    for epoch in range(epochs):

        activations, probs = forward(i_positions,i_layers)
        loss = -np.sum(np.log(probs[range(num_examples), i_labels])) / num_examples


        grads = backward(i_positions, i_labels, activations, probs, i_layers)

        for i, (dW, db) in enumerate(grads):
            i_layers[i].weight -= learning_rate * dW
            i_layers[i].bias -= learning_rate * db

        if SHOW_PROGRESS_AREAS and epoch % 100 == 0:
            plotAreas(i_positions, i_labels, i_layers, epoch)
        if SHOW_PROGRESS_GRAPHIS and epoch % 100 == 0:
            plotData(end_plot_points)
        if epoch % 100 == 0 and epoch > 0:
            print(f"Epoch {epoch}, loss: {loss:.4f}")
            end_plot_points.append((epoch, loss))

    activations, probs = forward(i_positions, i_layers)
    loss = -np.sum(np.log(probs[range(num_examples), i_labels])) / num_examples
    print(f"Epoch {epochs}, loss: {loss:.4f}")
    end_plot_points.append((epochs, loss))

# Plot areas
def plotAreas(i_position, i_label, i_layers, i_epoch):
    texelSize = 0.01
    x_min, x_max = i_position[:, 0].min() - 1, i_position[:, 0].max() + 1
    y_min, y_max = i_position[:, 1].min() - 1, i_position[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, texelSize),
                         np.arange(y_min, y_max, texelSize))
    grid = np.c_[xx.ravel(), yy.ravel()]
    _, probs = forward(grid, i_layers)
    Z = np.argmax(probs, axis=1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.4)
    plt.scatter(i_position[:, 0], i_position[:, 1], c=i_label, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title("Area Classification")
    plt.savefig(f"areas/a_e{i_epoch}.png")
    plt.show()

# Plot data
def plotData(data):
    x_vals = [p[0] for p in data]
    y_vals = [p[1] for p in data]

    # Trace
    plt.plot(x_vals, y_vals)
    plt.title("Loss evolution during training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"graphs/g_e{x_vals[len(x_vals)-1]}.png")
    plt.show()

def delete_folder(path):
    if not os.path.exists(path):
        return

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

    os.rmdir(path)

#____________________________________





# -------------------------------
# MAIN
print("Creating Folders...", end="")
delete_folder("graphs")
delete_folder("areas")
os.makedirs("areas", exist_ok=True)
os.makedirs("graphs", exist_ok=True)
print(" Done")


print("Creating Layers...", end="")
layers = []
# input -> hidden layer
Ws = 0.01 * np.random.randn(DIMENSION, HIDDEN_LAYER_SIZE)
bs = np.zeros((1, HIDDEN_LAYER_SIZE))
layers.append(Layer(Ws, bs))

for i in range(HIDDEN_LAYER_COUNT):
    W = 0.01 * np.random.randn(HIDDEN_LAYER_SIZE,HIDDEN_LAYER_SIZE)
    b = np.zeros((1, HIDDEN_LAYER_SIZE))
    layers.append(Layer(W, b))

# hidden -> output layer
We = 0.01 * np.random.randn(HIDDEN_LAYER_SIZE, NUMBER_OF_CLASSES)
be = np.zeros((1, NUMBER_OF_CLASSES))
layers.append(Layer(We, be))
print(" Done")



positions, labels = generate_data()
train(positions, labels, layers, EPOCHS, LEARNING_RATE)
plotAreas(positions, labels, layers,EPOCHS)
plotData(end_plot_points)




# -------------------------------
