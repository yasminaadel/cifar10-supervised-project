import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load data
# Training data is 50000 images, and 10000 test data totalling 60000 image
# there are also 10 classes so each class has 6000 images
(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()
X_train.shape

y_train.shape

# data in 2D array
y_train[:5]

# reshape from 2D to 1D
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)
y_train[:5]

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))  # Decrease image size
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])

plot_sample(X_train, y_train, 0)

plot_sample(X_train, y_train, 1)

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# split data
from sklearn.model_selection import train_test_split
X_train,X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#Artificial Neural Network
#Two dense layers with 3000 and 1000 neurons, no dropouts
#hyperparameters are ReLU for hidden layers and softmax for the output layer, optimizer is Stochastic Gradient Descent
ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)

#simplest architecture, less suitable for handling complex image data compared to convolutional architectures
ann.summary()

#Convolutional Neural Network
#Includes 2 convultional layers, 2 maxpool layers, a flattening layer leading to a dense layer with 64 neurons and softmax layer with 10 output classes
cnn1 = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#changed optimizer to adam
cnn1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn1.fit(X_train, y_train, epochs=10)

#Simple(basic feature extraction), used when computational resources are limited or when the image features are simple
cnn1.summary()

#adding dropout layers after each max pooling and before the final dense layer
#the flattening layer has a larger dense layer with 256 neurons, a dropout, and a softmax output layer
#increased complexity compared to CNN1 due to more dropout layers and larger dense layer
cnn2 = models.Sequential([
        layers.Conv2D(filters=32,kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),#technique to prevent overfitting
        layers.Conv2D(filters=64,kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

cnn2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn2.fit(X_train, y_train, epochs=10)

#used for datasets where there is a moderate risk of overfitting and more complex feature relationships
cnn2.summary()

#added an extra convolutional layers each conv layer followed by a max pooling and a dropout layer
#flattening layer leading into a more dense layer than the previous arch with 512 neurons
#deeper than CNN1 and CNN2
cnn3 = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

cnn3.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn3.fit(X_train, y_train, epochs=10)

#used for more complex tasks when capturing hierarchical features for performance
#suitable for larger datasets
cnn3.summary()

#increased filter sizes(5x5)
#similar to CNN3 in terms of depth but uses larger filter sizes
cnn4 = models.Sequential([
    layers.Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Conv2D(64, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

#changed optimizer to SGD
cnn4.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#used to capture more features details in one convolutional step
cnn4.fit(X_train, y_train, epochs=10)

cnn4.summary()

def predict(model, X_test, index):
    image = X_test[index]  #select the image by index
    image = image.reshape(1, 32, 32, 3)  #reshape the image to match the input shape of the model

    prediction = model.predict(image)  # Predict the class
    predicted_class = np.argmax(prediction)

    print("Predicted class:", classes[predicted_class])  #print the class name using the 'classes' list

# use different image indexes in predictions
print("Predictions using ANN model: ")
predict(ann, X_test, 0)
predict(ann, X_test, 15)
predict(ann, X_test, 100)
print(" ")

print("Predictions using First CNN model: ")
predict(cnn1, X_test, 0)
predict(cnn1, X_test, 15)
predict(cnn1, X_test, 100)
print(" ")

print("Predictions using Second CNN model: ")
predict(cnn2, X_test, 0)
predict(cnn2, X_test, 15)
predict(cnn2, X_test, 100)
print(" ")

print("Predictions using Third CNN model: ")
predict(cnn3, X_test, 0)
predict(cnn3, X_test, 15)
predict(cnn3, X_test, 100)
print(" ")

print("Predictions using Fourth CNN model: ")
predict(cnn4, X_test, 0)
predict(cnn4, X_test, 15)
predict(cnn4, X_test, 100)

#evaluate the result of all 5 architectures
test_loss_ann, test_acc_ann = ann.evaluate(X_test, y_test)
test_loss_cnn1, test_acc_cnn1 = cnn1.evaluate(X_test, y_test)
test_loss_cnn2, test_acc_cnn2 = cnn2.evaluate(X_test, y_test)
test_loss_cnn3, test_acc_cnn3 = cnn3.evaluate(X_test, y_test)
test_loss_cnn4, test_acc_cnn4 = cnn4.evaluate(X_test, y_test)

#dictionary to store models and their test accuracies and losses
model_performance = {
    "ANN Model": (test_acc_ann, test_loss_ann),
    "CNN1 Model": (test_acc_cnn1, test_loss_cnn1),
    "CNN2 Model": (test_acc_cnn2, test_loss_cnn2),
    "CNN3 Model": (test_acc_cnn3, test_loss_cnn3),
    "CNN4 Model": (test_acc_cnn4, test_loss_cnn4)
}

#determine the model with the best accuracy
best_model = max(model_performance, key=lambda x: model_performance[x][0])

#print out the accuracy and loss for comparison
print("\nModel Performance on Test Data:")
for model, (accuracy, loss) in model_performance.items():
    print(f"{model} - Accuracy: {accuracy*100:.2f}%, Loss: {loss:.4f}")

#print the best model based on accuracy
best_accuracy, best_loss = model_performance[best_model]
print(f"\nThe best model based on accuracy is {best_model} with an accuracy of {best_accuracy*100:.2f}% and a loss of {best_loss:.4f}.")