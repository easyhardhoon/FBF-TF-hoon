import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


print("before : ",x_train.shape[0])
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
#x_train = x_train.astype('int8')
#x_test = x_test.astype('int8')
#x_test = x_test.astype('int8')
x_test = x_test.astype('float32')
x_train = x_train.astype('float32')

#print(x_test[0])
x_train = x_train / 255
x_test = x_test / 255

print('x_train shape:', x_train.shape)

#--------------------------------------------------
# Define input shape
input_shape = (28, 28, 1)

# Define input layer
inputs = Input(shape=input_shape)

# Define split layer using Lambda layer with custom function
def split_layer(x):
    return tf.split(x, num_or_size_splits=2, axis=2)

split = Lambda(split_layer)(inputs)

# Define path 1
path1 = Conv2D(32, (3, 3), activation='relu')(split[0])
path1 = MaxPooling2D((2, 2))(path1)
path1 = Flatten()(path1)
path1 = Dense(128, activation='relu')(path1)

# Define path 2
path2 = Conv2D(32, (3, 3), activation='relu')(split[1])
path2 = MaxPooling2D((2, 2))(path2)
path2 = Flatten()(path2)
path2 = Dense(128, activation='relu')(path2)

# Merge paths
merged = tf.keras.layers.concatenate([path1, path2])

# Define output layer
outputs = Dense(10, activation='softmax')(merged)

# Define model
model = Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test)

print('accuacy :', test_acc)

model.save('saved_model/mnist_8')
