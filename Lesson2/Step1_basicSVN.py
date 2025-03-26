import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.datasets import cifar10
import time
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Defines
SHOW_IMAGES = False # slow if True
SHOW_MESSAGES = True
BEST_MATCH = 1 # 2 : comapare with the two best matches, and so on
EPOCH = 5

# Label's dictionary
cifar10_classes = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

K, D = 10, 3072  # 10 classes, 3072 features


def display_images(image, predicted_label, real_label):

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(image)
    ax[0].axis("off")
    ax[0].set_title(f"Test: {cifar10_classes[real_label]} ({real_label})")
    if predicted_label == real_label:
        ax[1].text(0.5, 0.6, "Success", fontsize=12, ha="center", va="center", color="green", fontweight="bold")
    else:
        ax[1].text(0.5, 0.6, "Fail", fontsize=12, ha="center", va="center", color="red", fontweight="bold")
    ax[1].text(0.5, 0.5, f"Predicted label : {cifar10_classes[predicted_label]} ({predicted_label})", fontsize=12, ha="center", va="center")
    ax[1].axis("off")
    plt.show()


if __name__ == "__main__":
    startTime = time.time()
    process_train_images = train_images / 127.5 - 1.
    process_test_images = test_images / 127.5-1.

    W = np.random.randn(K, D) * 0.001
    delta = 1.1
    sigma = 0.01

    try:
        print(f"Training model... with delta = {delta}, sigma = {sigma}")
        for epoch in range(EPOCH):
            print("Epoch {}".format(epoch))
            for i in range(len(process_train_images)):
                image_vector = process_train_images[i].flatten()
                scores = W @ image_vector
                real_label = train_labels[i][0]
                real_label_score = scores[real_label]
                difference = 0
                for j in range(len(scores)):
                    if j != real_label:
                        other_label_score = scores[j]
                        difference += max(0, other_label_score - real_label_score + delta)

                if difference > 0:
                    W[real_label] = W[real_label] + sigma * image_vector
                    for  j in range(len(scores)):
                        if j != real_label:
                            W[j] = W[j] - sigma * image_vector

        elapsed_time = time.time() - startTime
        elapsed_minutes = int(elapsed_time // 60)
        elapsed_seconds = elapsed_time - 60 * elapsed_minutes
        print(f"Training done in : {elapsed_minutes}min:{elapsed_seconds:.1f}sec")




        success_count= 0
        for i in range(len(test_images)):
            image_vector = process_test_images[i].flatten()
            scores = W @ image_vector
            sorted_scores = np.argsort(scores)[::-1]

            for j in range(BEST_MATCH):
                label_pred = sorted_scores[j]
                if SHOW_IMAGES:
                    display_images(test_images[i], label_pred, test_labels[i][0])
                if SHOW_MESSAGES:
                    print("sucess" if label_pred == test_labels[i][0] else "fail", "  predicted label:", label_pred, " real label:", test_labels[i][0])
                if label_pred == test_labels[i][0]:
                    success_count += 1
                    break


        #for i in range(K):
        #    min_val = W[i].min()
        #    max_val = W[i].max()
        #    normalized_image = (W[i] - min_val) / (max_val - min_val)
        #    display_images(normalized_image.reshape(32, 32, 3), i, i)


        print("success ratio:", success_count/(len(test_images)) * 100., "%")


    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    elapsed_time = time.time() - startTime
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = elapsed_time - 60 * elapsed_minutes
    print(f"Done in : {elapsed_minutes}min:{elapsed_seconds:.1f}sec")


