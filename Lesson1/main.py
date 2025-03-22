import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
import os

from keras.src.metrics.accuracy_metrics import accuracy

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.datasets import cifar10

from queue import Empty
import time
from collections import Counter

#defines
USE_L1        = False           # make sure L1 or L2 is true
SHOW_PLOTS    = True
SHOW_MATCHES  = False
SHOW_MESSAGES = False
MAX_K = 10


# Load the dataset CIFAR-10
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Label's dictionary
cifar10_classes = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}


def l1_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def l2_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# start, end for the training data
# o_totalCount of processed test
# o_matchCount of succeed test
# lock for the multiprocessing
# o_queue to stock image data
# compare images form x_test to x_train
def process_data(start, end, o_total_count, o_match_count, lock, o_queue):
    with lock:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    try:
        for testI in range(start, end):
            test_img = x_test[testI]
            test_label = y_test[testI][0]

            distances = [(l1_distance(test_img, x_train[i]) if USE_L1 else l2_distance(test_img, x_train[i]) , y_train[i][0], i) for i in range(len(x_train))]

            distances.sort(key=lambda x: x[0])



            with lock:
                o_total_count.value += 1
                for k in range(1, MAX_K + 1):
                    k_nearest_neighbors = distances[:k]

                    # Vote majoritaire sur les K voisins
                    k_labels = [neighbor[1] for neighbor in k_nearest_neighbors]
                    predicted_label = Counter(k_labels).most_common(1)[0][0]
                    if predicted_label  == test_label:
                        o_match_count[k-1] += 1
                        if SHOW_MATCHES:
                            for neighbor in range(k):
                                nLabel = k_labels[neighbor]
                                if nLabel == predicted_label:
                                    o_queue.put((testI, k_nearest_neighbors[neighbor][2]))
                                    break

                    if SHOW_MESSAGES:
                        print(k," " if testI <1000 else ""," " if testI <100 else ""," " if testI <10 else "", testI, f" - Success rate:{round(o_match_count[k-1] / o_total_count.value * 100.0, 2)}% \t - Total count :", o_total_count.value, "/", len(x_test),f"({round(o_total_count.value/len(x_test) * 100.0,1)}% done)"
                          "\t  - Match count :", o_match_count[k-1])
                if SHOW_MESSAGES:
                    print("\n")

    except KeyboardInterrupt:
        return 0

    return 0

def display_images(queue, total_test_samples, lock, matchCount, totalCount):
    with lock:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    try :
        while totalCount.value < total_test_samples:
            if SHOW_MATCHES:
                testI, closestI = queue.get(timeout=60)
                test_img = x_test[testI]
                test_label = y_test[testI][0]
                closest_img = x_train[closestI]
                closest_label = y_train[closestI][0]

                fig, ax = plt.subplots(1, 2, figsize=(6, 3))
                ax[0].imshow(test_img)
                ax[0].set_title(f"Test: {cifar10_classes[test_label]} ({test_label})")
                ax[1].imshow(closest_img)
                ax[1].set_title(f"Match: {cifar10_classes[closest_label]} ({closest_label})")
                info_text = "Success" if closest_label == test_label else "Fail"
                plt.figtext(0.5, 0.005, info_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
                plt.show()

            if SHOW_PLOTS and totalCount.value > 0:
                x = np.arange(1, MAX_K + 1)
                y = np.frombuffer(matchCount.get_obj(), dtype=np.int32)
                y = y / totalCount.value
                fig, ax = plt.subplots()
                ax.plot(x, y, marker='o', linestyle='-')
                ax.set_ylim(min(y) - 0.01, max(y) + 0.01)
                ax.set_xlabel("K-NN")
                ax.set_ylabel("Match count (normalized)")
                ax.set_title(f"{'L1' if USE_L1 else 'L2'} after {totalCount.value} iterations")
                plt.xticks(x)
                plt.grid(True)
                plt.show()

    except KeyboardInterrupt:
        return 0
    except Empty:
        print("Image queue is empty ", totalCount.value, "/",total_test_samples )



if __name__ == "__main__":
    processes = []
    startTime = time.time()  # Démarre le chrono
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    try:
        multiprocessing.freeze_support()  # Needed on windows

        num_processes = max(1,os.cpu_count()-2)
        total_test_samples = len(x_test)
        chunk_size = total_test_samples // num_processes

        # shared variables
        totalCount = multiprocessing.Value("i", 0)
        matchCount = multiprocessing.Array("i", [0] * MAX_K)
        lock = multiprocessing.Lock()
        queue = multiprocessing.Queue()

        print("Available processes: ", num_processes)
        print("Test size: ", len(x_test))



        #for k in range(1): #range(1, MAX_K+1):
        matchCount.value = 0
        totalCount.value = 0
        #print("===== With ", k, "-NN")
        for i in range(num_processes):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_processes - 1 else total_test_samples
            p = multiprocessing.Process(target=process_data, args=(start_idx, end_idx, totalCount, matchCount, lock, queue))
            processes.append(p)
            p.start()


        display_process = multiprocessing.Process(target=display_images, args=(queue,total_test_samples, lock, matchCount, totalCount))
        processes.append(display_process)
        display_process.start()

        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    elapsed_time = time.time()-startTime
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = elapsed_time - 60 * elapsed_minutes
    print(f"Done in : {elapsed_minutes}min:{elapsed_seconds:.1f}sec")
