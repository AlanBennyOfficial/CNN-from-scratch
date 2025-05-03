import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
from sklearn.datasets import fetch_openml

print("Downloading MNIST from OpenML...")
mnist = fetch_openml('mnist_784', version=1)
images = mnist.data.values.reshape(-1, 28, 28).astype(np.uint8)
labels = mnist.target.astype(int).values  # ensure labels are integers

test_images = images[:10000]
test_labels = labels[:10000]
train_images = images[10000:]
train_labels = labels[10000:]

conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10)   # 13x13x8 -> 10

def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''
    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

def train(im, label, lr=.005):
    '''
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    '''
    out, loss, acc = forward(im, label)

    # Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # Backprop
    gradient = softmax.backprop(gradient, lr)
    # TODO: backprop MaxPool2 layer
    # TODO: backprop Conv3x3 layer

    return loss, acc

print('MNIST CNN initialized!')

# Train!
loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i % 100 == 99:
        print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (i + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0

    l, acc = train(im, label)   
    loss += l
    num_correct += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)