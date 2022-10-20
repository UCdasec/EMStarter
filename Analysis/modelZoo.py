import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization
from tensorflow.keras.layers import GlobalMaxPool1D, Input, AveragePooling1D
from tensorflow.keras.layers import Flatten, GlobalMaxPooling1D
from tensorflow.keras.layers import Activation, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Lambda, Dot
from tensorflow.keras.callbacks import CSVLogger


### CNN Best model
def cnn_best2(input_shape, classes=256):
    # From VGG16 design
    # input_shape = (1400,1)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv1D(64, 11, strides=2, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # Block 2
    x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    # Block 4
    x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    # Block 5
    x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn_best2')
    optimizer = RMSprop(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def triplet_cnn(input_shape=None, emb_size=None):
    '''
    This generates the embeddings required for the triplet network
    '''
    input_data = Input(shape=input_shape)

    # Block 1
    x = Conv1D(64, 11, strides=2, activation='relu', padding='same', name='block1_conv1')(input_data)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # Block 2
    x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    # Block 4
    x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    # Block 5
    x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    # x = Dense(4096, activation='relu', name='fc1')(x)
    # x = Dense(4096, activation='relu', name='fc2')(x)
    # x = Dense(classes, activation='softmax', name='predictions')(x)

    dense_layer = Dense(emb_size, name='feature_vector')(x)
    # Create model.
    shared_conv = Model(inputs=input_data, outputs=dense_layer)
    return shared_conv