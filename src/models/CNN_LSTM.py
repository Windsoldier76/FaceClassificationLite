from keras import layers
from keras.layers import Activation, Dropout, Conv2D, Dense
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten, LSTM, Input, MaxPooling2D
from keras.models import Model
from keras.regularizers import l2
# input shape is 128*128*16
def CNN(input_shape, num_classes):
    # todo: 1. data process  2. patameters 3. train


    #base
    img_input = Input(input_shape)
    x = Conv2D(32, (7, 7), strides=(1, 1), padding='same')(img_input)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)


    #module 1
    residual = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1),
                      padding='same')(x)
    residual = Activation("relu")(x)

    x = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1),
               padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1),
               padding='same')(x)
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x = Activation("relu")(x)


    # module 2
    residual = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                      padding='same')(x)
    residual = Activation("relu")(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               padding='same')(x)
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x = Activation("relu")(x)


    # module 3
    residual = x

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               padding='same')(x)
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x = Activation("relu")(x)


    # module 4
    residual = x

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               padding='same')(x)
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x = Activation("relu")(x)


    # module 5
    residual = x

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               padding='same')(x)
    x = Activation("relu")(x)

    x = layers.add([x, residual])
    x = Activation("relu")(x)

    # module 6
    residual = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                      padding='same')(x)
    residual = Activation("relu")(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
               padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
               padding='same')(x)
    x = Activation("relu")(x)

    x = layers.add([x, residual])
    x = Activation("relu")(x)


    # module 7
    residual = x

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
               padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
               padding='same')(x)
    x = Activation("relu")(x)

    x = layers.add([x, residual])
    x = Activation("relu")(x)


    # module 8
    residual = x

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
               padding='same')(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
               padding='same')(x)
    x = Activation("relu")(x)

    x = layers.add([x, residual])
    x = Activation("relu")(x)

    # flatten
    x = Flatten()(x)

    # LSTM layer
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(256, return_sequences=False)(x)

    x = Dense(6)(x)
    output = Activation('softmax', name='predictions')(x)
    model = Model(img_input, output)
    return model
