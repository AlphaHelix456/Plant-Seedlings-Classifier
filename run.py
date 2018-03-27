from image_preprocessing import load_train, extract_labels, extract_features
from model import train_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import load_model

IMAGE_SIZE = 66

def plot_confusion_matrix(X, Y, cmap=plt.cm.Greens, figsize=(10, 6)):
    Y_pred = model.predict(X)
    Y_pred = np.argmax(Y_pred, axis=1)

    Y_true = np.argmax(Y, axis=1)
    cm = confusion_matrix(Y_true, Y_pred)
    plt.figure(figsize=figsize)
    ax = sns.heatmap(cm, cmap=cmap, annot=True, square=True)
    plt.show()
    

if __name__ == '__main__':
    train = load_train()
    X_train = extract_features(train, IMAGE_SIZE)
    Y_train = extract_labels(train)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                      test_size=0.1, 
                                                      random_state=42)

    # train_model(X_train, X_val, Y_train, Y_val, IMAGE_SIZE)

    model = load_model('model.h5')

    # Change X_val and Y_val to evaluate different data
    loss, accuracy = model.evaluate(X_val, Y_val)
    print('Final Loss: {}, Final Accuracy: {}'.format(loss, accuracy))

    plot_confusion_matrix(X_val, Y_val)
