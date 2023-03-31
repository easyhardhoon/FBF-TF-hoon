import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


print("before : ",x_train.shape[0])
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)

#x_train = x_train.astype('int8')
#x_test = x_test.astype('int8')
#x_test = x_test.astype('int8')
x_test = x_test.astype('float32')
x_train = x_train.astype('float32')

#print(x_test[0])
x_train = x_train / 255
x_test = x_test / 255

print('x_train shape:', x_train.shape)

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input,Conv2D, Dropout, Flatten, MaxPooling2D, concatenate,Add, Multiply 
from tensorflow.keras.utils import plot_model


#
##NOTE 3-times CONV & dynamic concatenate  MNIST Model by Model class
#
## ---
#class model_GPU(Model):
#    def __init__(self):
#        super(model_GPU, self).__init__()
#        self.conv1 = Conv2D(10,(3,3),activation=tf.nn.leaky_relu)
#        self.pool1 = MaxPooling2D(2,2)
#        self.conv2 = Conv2D(20,(3,3),activation=tf.nn.leaky_relu)
#        self.pool2 = MaxPooling2D(2,2)
#        self.conv3 = Conv2D(40,(3,3),activation=tf.nn.leaky_relu)
#        self.flat = Flatten()
#        self.dense1 = Dense(128, activation=tf.nn.leaky_relu)
#        self.dense2 = Dense(10, activation = tf.nn.softmax)
#        self.dummy1 = Conv2D(1,(3,3), trainable= False)
#        self.dummy2 = Conv2D(1,(3,3), trainable= False)
#        self.dummy3 = Conv2D(1,(3,3), trainable= False)
#    def call(self, x):
#        model_dummy_1 = self.dummy1(tf.zeros([1,28,28,1]))
#        model_dummy_1.trainable =  False
#        net = self.conv1(x)
#        net = concatenate([net,model_dummy_1], axis=3)
#        net = self.pool1(net)
#        net = self.conv2(net)
#        model_dummy_2 = self.dummy2(tf.zeros([1,13,13,1]))
#        model_dummy_2.trainable =  False
#        net = concatenate([net,model_dummy_2],axis=3)
#        net = self.pool2(net)
#        net = self.conv3(net)
#        model_dummy_3  = self.dummy3(tf.zeros([1,5,5,1]))
#        model_dummy_3.trainable =  False
#        net = concatenate([net,model_dummy_3], axis=3)
#        net = self.flat(net)
#        net = self.dense1(net)
#        net = self.dense2(net)
#        return net
#    def model(self):
#        x = Input(shape=input_shape)
#        return Model(inputs=[x], outputs=self.call(x))
#
#myGpuModel = model_GPU().model()
#myGpuModel.build([1,28,28,1])
#myGpuModel.summary()
#plot_model(myGpuModel,to_file = 'model_gpu.png',show_shapes=True)
#myGpuModel.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])
#
#
#
#myGpuModel.fit(x=x_train,y=y_train, batch_size=1, epochs=5)
#
#test_loss, test_acc = myGpuModel.evaluate(x_test, y_test, batch_size=1)
#
#print('accuacy :', test_acc)
#
#myGpuModel.save('saved_model/11')
#
#--------------------------------------------------------------------------------------
#NOTE just 3-times CONV MNIST Model by sequential class
#input_shape = (1,28,28,1)
#model = Sequential()
#model.add(Input(shape=input_shape))
#model.add(Conv2D(5, kernel_size=(3,3), activation=tf.nn.elu))
#model.add(Conv2D(5, kernel_size=(3,3),activation=tf.nn.elu))
#model.add(Conv2D(5, kernel_size=(3,3), input_shape=input_shape, activation=tf.nn.elu))
#model.add(Add())
#model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(20, kernel_size=(3,3), activation=tf.nn.elu))
#model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(40, kernel_size=(3,3), activation = tf.nn.elu))
#model.add(Dropout(0.2))
#model.add(Flatten())
#model.add(Dense(128, activation=tf.nn.relu))
#model.add(Dense(10,activation=tf.nn.softmax))
#model.add(Multiply())  #++


#----------------------------------------------------
#model VGG_16
model = Sequential()
model.add(tf.keras.layers.Conv2D(input_shape = input_shape, filters = 16, kernel_size= (3,3), strides= (1,1), activation= 'relu', padding = 'same'))
model.add(tf.keras.layers.Conv2D(filters= 16, kernel_size= (3,3), strides= (1,1), activation= 'relu', padding= 'same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size= (2,2)))

model.add(tf.keras.layers.Conv2D(filters= 32, kernel_size= (3,3), strides= (1,1), activation= 'relu', padding= 'same'))
model.add(tf.keras.layers.Conv2D(filters= 32, kernel_size= (3,3), strides= (1,1), activation= 'relu', padding= 'same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size= (2,2)))

model.add(tf.keras.layers.Conv2D(filters= 64, kernel_size= (3,3), strides= (1,1), activation= 'relu', padding= 'same'))
model.add(tf.keras.layers.Conv2D(filters= 64, kernel_size= (3,3), strides= (1,1), activation= 'relu', padding= 'same'))
model.add(tf.keras.layers.Conv2D(filters= 64, kernel_size= (3,3), strides= (1,1), activation= 'relu', padding= 'same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size= (2,2)))

model.add(tf.keras.layers.Conv2D(filters= 128, kernel_size= (3,3), strides= (1,1), activation= 'relu', padding= 'same'))
model.add(tf.keras.layers.Conv2D(filters= 128, kernel_size= (3,3), strides= (1,1), activation= 'relu', padding= 'same'))
model.add(tf.keras.layers.Conv2D(filters= 128, kernel_size= (3,3), strides= (1,1), activation= 'relu', padding= 'same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size= (2,2)))

model.add(tf.keras.layers.Conv2D(filters= 128, kernel_size= (3,3), strides= (1,1), activation= 'relu', padding= 'same'))
model.add(tf.keras.layers.Conv2D(filters= 128, kernel_size= (3,3), strides= (1,1), activation= 'relu', padding= 'same'))
model.add(tf.keras.layers.Conv2D(filters= 128, kernel_size= (3,3), strides= (1,1), activation= 'relu', padding= 'same'))
#model.add(tf.keras.layers.MaxPooling2D(pool_size= (2,2)))  #HOON

#fully connected
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))   #HOON 
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=5) #10 .. 

test_loss, test_acc = model.evaluate(x_test, y_test)

print('accuacy :', test_acc)

model.save('saved_model/mnist_10')

#--------------------------------------------------------------------------------------
