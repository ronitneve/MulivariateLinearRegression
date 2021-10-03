import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# importing the Iris data.
df = pd.read_csv("iris.csv", names=[
                 "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"])

# Random shuffle the DataFrame so that the Train and Test set has _Similar_ data points.
Sdf = df.sample(frac=1, ignore_index=True)
trainDF = Sdf[:115]
testDF = Sdf[115:]

Train_row, Train_col = trainDF.shape
Test_row, Test_col = testDF.shape
mapping = {
    'Iris-setosa': 1,
    'Iris-versicolor': 2,
    'Iris-virginica': 3
}
# Prepping Target and Feature values.
Test_X = testDF.drop(['Species'], axis=1).values
Test_X = np.hstack(((np.ones((Test_row, 1))), Test_X))
# encode all Target values to numericals.
Test_y = testDF.Species.replace(mapping).values.reshape(Test_row, 1)
Train_X = trainDF.drop(['Species'], axis=1).values
Train_X = np.hstack(((np.ones((Train_row, 1))), Train_X))
Train_y = trainDF.Species.replace(mapping).values.reshape(Train_row, 1)

# Setting initial values of weight randomly
weight = np.random.randn(1, 5)
print("Initial weights : %s" % (weight))
epochs = 100000
alpha = 0.001

# Matrix to keep track fo mean squared error through all the epochs
mse = np.zeros(epochs)

for i in range(epochs):
    mse[i] = (1/(2 * Train_row) *
              np.sum((np.dot(Train_X, weight.T) - Train_y) ** 2))
    weight -= np.dot((np.dot(Train_X, weight.T) - Train_y).reshape(1,
                     Train_row), Train_X) * (alpha/Train_row)

prediction = np.round(np.dot(Test_X, weight.T))
Accuracy = np.sum(np.equal(Test_y, prediction)) / len(prediction)
print("Updated weights : %s" % (weight))
print("Accuracy of the model: {0}".format(Accuracy))


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
fig.tight_layout(pad=3.0)
# Plot the mean squared error
ax1.set_ylim([0, 0.15])
ax1.set_title("Error")
ax1.plot(np.arange(epochs), mse)

# Plot the predicted and orignal target values
ax2.scatter(np.arange(35), Test_y, color='red',
            label='Orignal Value', alpha=0.5)
ax2.scatter(np.arange(35), prediction, color='blue',
            label='Predicted Value', s=10)
ax2.set(xlabel="Dataset size", ylabel="Iris Flower (1-3)",
        title="Predictions (Iris-setosa = 1, Iris-versicolor = 2, Iris-virginica = 3)")
ax2.legend()
plt.show()
