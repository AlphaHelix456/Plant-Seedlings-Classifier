from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, \
     BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

BATCH_SIZE = 32
EPOCHS = 75

# Experimentation goals:
# LeakyReLU activation
# Adding Dropout layers
# Different optimizers (i.e. RMSprop, Adagrad, Adadelta)
# Filter values for convolutional layers

def construct_model(image_size):
    model = Sequential()
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                     input_shape=(image_size, image_size, 3), activation='relu'))
    model.add(BatchNormalization()) 
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten()) 
    model.add(Dense(256, activation='relu'))
    model.add(Dense(12, activation='softmax'))
    
    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    
    model.summary()
    return model


def train_model(X_train, X_val, Y_train, Y_val, image_size):
    model = construct_model(image_size)
    # ReduceLROnPlateau and ModelCheckpoint
    callbacks = get_callbacks('model.h5')
    # Generates batches of image data with data augmentation
    datagen = get_datagen(X_train)
    # Fits the model on batches with real-time data augmentation
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                   steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                   epochs=EPOCHS,
                   verbose=2,
                   callbacks=callbacks,
                   validation_data=(X_val, Y_val))


def get_callbacks(file):
    annealer = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5,
                                 verbose=1, min_lr=1e-5)
    checkpoint = ModelCheckpoint(file, verbose=1, save_best_only=True)

    return [annealer, checkpoint]


def get_datagen(X_train):
    datagen = ImageDataGenerator(rotation_range=360, # Degree range for random rotations
                            width_shift_range=0.2, # Range for random horizontal shifts
                            height_shift_range=0.2, # Range for random vertical shifts
                            zoom_range=0.2, # Range for random zoom
                            horizontal_flip=True, # Randomly flip inputs horizontally
                            vertical_flip=True) # Randomly flip inputs vertically
    
    datagen.fit(X_train)
    
