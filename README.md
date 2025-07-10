# CIFAR-10 Image Classification

A supervised learning project focused on *image classification using the CIFAR-10 dataset. This work compares the performance of **classical machine learning algorithms* and *deep learning models* (ANN and CNN architectures) using extracted features.

## Dataset

- *CIFAR-10: 60,000 color images of size 32×32 across **10 classes*
  - Training: 50,000 images
  - Testing: 10,000 images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Project Objectives

- Apply *feature extraction* using pretrained CNNs
- Compare multiple *classical ML classifiers*
- Train and evaluate multiple *ANN and CNN architectures*
- Identify the *best-performing model* based on test accuracy

## Algorithms & Models Used
### Deep Learning:
#### ANN:
- Flatten input → Dense(3000) → Dense(1000) → Dense(10, softmax)
- Optimizer: *SGD*, Loss: sparse_categorical_crossentropy

#### CNN Architectures:
1. *CNN1*: 2 Conv2D + MaxPooling → Dense(64)
2. *CNN2*: Dropouts + Dense(256)
3. *CNN3*: 3 Conv2D blocks + Dense(512)
4. *CNN4*: 5×5 filters + SGD optimizer + Dense(512)

All CNNs use *ReLU* activations and *softmax output*.


## 📈 Algorithm Comparison Chart

<img width="855" height="523" alt="image" src="https://github.com/user-attachments/assets/6356f3c8-1f30-4e47-b304-758fc81747ae" />

| Algorithm        | Accuracy (%) |
|------------------|--------------|
| Logistic Regression | 97 |
| KNN                | 96 |
| Neural Network     | 98 |
| Naive Bayes        | 78 |
| Decision Tree      | 88 |
| Random Forest      | 97 |
| SVM                | 0 |
