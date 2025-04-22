import numpy as np
import matplotlib.pyplot as plt
import os

import multiprocessing
from multiprocessing import Pool, cpu_count


#____________________________________
# DEFINES
POINT_PER_CLASS = 100
NUMBER_OF_CLASSES = 3
SPIRAL_SHAPE = True
SHOW_PROGRESS_GRAPHICS = True
SHOW_SUCCESS = False
TRAINING = False
EPOCHS = 50000
LOSS_TARGET = 0.1
LEARNING_RATE = 0.2
HIDDEN_LAYER_SIZE = 256
HIDDEN_LAYER_COUNT = 2 # there is 1 layer by default even when this is 0
#____________________________________

class Layer:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

end_plot_points = []

#____________________________________
# FUNCTIONS

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    y_train = y_train.flatten()
    x_test = x_test.astype('float32') / 255.0
    y_test = y_test.flatten()
    return x_train, y_train, x_test, y_test

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

    global end_plot_points
    num_examples = i_positions.shape[0] #POINT_PER_CLASS*NUMBER_OF_CLASSES

#    plotAreas(positions, labels, i_layers,0)

    activations, probs = forward(i_positions, i_layers)
    loss = -np.sum(np.log(probs[range(num_examples), i_labels])) / num_examples
    print(f"Epoch {0}, loss: {loss:.4f}")
    #end_plot_points.append((0, loss))
    epoch =0
    lossAvg = 0
    bestLoss = np.inf
    while loss > LOSS_TARGET:

        activations, probs = forward(i_positions,i_layers)
        loss = -np.sum(np.log(probs[range(num_examples), i_labels])) / num_examples
        if bestLoss == np.inf:
            bestLoss = loss


        grads = backward(i_positions, i_labels, activations, probs, i_layers)

        for i, (dW, db) in enumerate(grads):
            i_layers[i].weight -= learning_rate * dW
            i_layers[i].bias -= learning_rate * db

        print(f"Epoch {epoch}, loss: {loss}")
        lossAvg += loss
        if SHOW_PROGRESS_GRAPHICS and epoch % 10 == 0:
            if epoch > 0:
                end_plot_points.append((epoch, lossAvg/10.0))
            else:
                end_plot_points.append((epoch, loss))
            lossAvg = 0.0
            end_plot_points = end_plot_points[-300:]
            plotData(end_plot_points)

        if loss < bestLoss:
            bestLoss = loss
            save_model(layers)


        epoch = epoch +1

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


def save_model(layers, filename=f"model_weights_{HIDDEN_LAYER_COUNT}_{HIDDEN_LAYER_SIZE}.npz"):
    weights = {}
    for i, layer in enumerate(layers):
        weights[f"W{i}"] = layer.weight
        weights[f"b{i}"] = layer.bias
    np.savez(filename, **weights)
    print(f"✅ Model saved to : {filename}")

def load_model(filename=f"model_weights_{HIDDEN_LAYER_COUNT}_{HIDDEN_LAYER_SIZE}.npz"):
    data = np.load(filename)
    layers = []
    i = 0
    while f"W{i}" in data:
        W = data[f"W{i}"]
        b = data[f"b{i}"]
        layers.append(Layer(W, b))
        i += 1
    print(f"📂 Model loaded from : {filename}")
    return layers
#____________________________________





# -------------------------------
# MAIN

if __name__ == "__main__":

    from keras.datasets import cifar10
    multiprocessing.freeze_support()  # Needed on windows
    lock = multiprocessing.Lock()
    num_processes = max(1, os.cpu_count() - 2)

    positions, labels, test_pos, test_label = load_cifar10()
    test_pos_show = test_pos
    positions = positions.reshape(len(positions), -1)  # Flatten: (N, 32, 32, 3) ➜ (N, 3072)
    test_pos  = test_pos.reshape(len(test_pos), -1)
    inputSize =  32 * 32 * 3  # 3072
    outputSize = 10



    total_test_samples = len(positions)
    chunk_size = total_test_samples // num_processes
    print("Available processes: ", num_processes)

    print("Creating Folders...", end="")
    delete_folder("graphs")
    delete_folder("areas")
    os.makedirs("areas", exist_ok=True)
    os.makedirs("graphs", exist_ok=True)
    print(" Done")

    layers = load_model()

    if TRAINING:
        train(positions, labels, layers, EPOCHS, LEARNING_RATE)
        save_model(layers)

    cifar10_classes = {
        0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
    }

    totalCount = 0
    successCount = 0

    for i in range(len(test_pos_show)):
        totalCount += 1
        factivations, fprobs = forward(test_pos[i], layers)
        classe_predite = np.argmax(fprobs)

        if classe_predite == test_label[i]:
            successCount += 1

            if SHOW_SUCCESS:
                top_indices = (np.argsort(fprobs)[0])[::-1]
                top_classes = [cifar10_classes[idx] for idx in top_indices]
                top_probs = [fprobs[0][idx] for idx in top_indices]

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

                ax1.imshow(test_pos_show[i])
                ax1.set_title(f"Real label: {cifar10_classes[test_label[i]]}")
                ax1.axis('off')
                result_text = "Success!" if classe_predite == test_label[i] else "Fail"
                result_color = "green" if classe_predite == test_label[i] else "red"
                ax1.text(0.5, -0.15, result_text, transform=ax1.transAxes, ha='center', fontsize=12, color=result_color)

                y_pos = np.arange(len(top_classes))
                ax2.barh(y_pos, top_probs[::-1], color='skyblue')
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(top_classes[::-1])
                ax2.set_xlim(0, 1)
                ax2.set_xlabel("Probability")
                ax2.set_title("Top predictions")

                plt.tight_layout()
                plt.show()

        print(f"Success ratio: {successCount}/{totalCount}   -   {successCount/totalCount} ")

# -------------------------------
