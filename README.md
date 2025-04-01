# AI - Stanford CS231n Implementation

## Overview

This repository contains an implementation of concepts and assignments from Stanford's renowned [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/) course. The goal is to provide a hands-on approach to deep learning and computer vision through practical exercises and visualizations.

## Lessons

---
### Lesson 2
#### Step 1

We trained a basic SVM model on CIFAR-10 and achieved an accuracy of approximately 32.15% using the selected hyperparameters: Δ = 1.1, η = 0.01, and 5 epochs.
By considering the top two predictions instead of only the highest-scoring class, the accuracy increases to 48.6%.


This represents an improvement from the first lesson ~27% accuracy to ~32%, meaning that a basic SVM performs ~18.5% better than in Lesson1.


#### Step2

With the Softmax approach, we achieve an accuracy of approximately 42%, which represents an improvement of around 30.6% compared to step 1.

Here is a recap of the progress so far:
<p align="center">
    <img src="/Lesson2/progress_lesson2.png" alt="progress_lesson2" width="45%" />
</p>

---
### Lesson 1

This lesson introduces fundamental concepts in image classification and deep learning, including:
- Understanding neural networks and their architecture
- Implementing basic image processing techniques
- Training a simple model for image recognition

#### Visualizations

<p align="center">
    <img src="/Lesson1/L1.png" alt="Lesson 1 Visualization 1" width="45%" />
    <img src="/Lesson1/L2.png" alt="Lesson 1 Visualization 2" width="45%" />
</p>

---
## Getting Started

### Prerequisites
To run the code in this repository, ensure you have the following dependencies installed:

````bash
pip install -r requirements.txt
````

### Usage
Clone the repository and navigate to the desired lesson:

````bash
git clone https://github.com/Noctiluce/cs231n.git
cd cs231n/Lesson1
python lesson1.py
````

## Contributing
Feel free to open issues or submit pull requests to improve the implementation.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Stay tuned for more lessons and updates!

