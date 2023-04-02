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
from tensorflow.keras.layers import Dense, Input,Conv2D, Dropout, Flatten, MaxPooling2D, concatenate 
from tensorflow.keras.utils import plot_model


#
##NOTE 3-times CONV & dynamic concatenate  MNIST Model by Model class
#
## ---
class model_VGG(Model):
    def __init__(self):
        super(model_VGG, self).__init__()
        self.conv1_1 = Conv2D(5,(3,3),strides= (1,1),activation=tf.nn.leaky_relu, padding='same')
        self.conv1_2 = Conv2D(5,(3,3), strides=(1,1), activation=tf.nn.leaky_relu, padding='same')
        self.pool1 = MaxPooling2D(2,2)
        self.conv2_1 = Conv2D(10,(3,3), strides=(1,1), activation=tf.nn.leaky_relu, padding='same')
        self.conv2_2 = Conv2D(10,(3,3), strides=(1,1), activation=tf.nn.leaky_relu, padding='same')
        self.pool2 = MaxPooling2D(2,2)
        self.conv3_1 = Conv2D(20,(3,3), strides=(1,1), activation=tf.nn.leaky_relu, padding='same')
        self.conv3_2 = Conv2D(20,(3,3), strides=(1,1), activation=tf.nn.leaky_relu, padding='same')
        self.conv3_3 = Conv2D(20,(3,3), strides=(1,1), activation=tf.nn.leaky_relu, padding='same')
        self.pool3 = MaxPooling2D(2,2)
        self.conv4_1 = Conv2D(30,(3,3), strides=(1,1), activation=tf.nn.leaky_relu, padding='same')
        self.conv4_2 = Conv2D(30,(3,3), strides=(1,1), activation=tf.nn.leaky_relu, padding='same')
        self.conv4_3 = Conv2D(30,(3,3), strides=(1,1), activation=tf.nn.leaky_relu, padding='same')
        self.pool4 = MaxPooling2D(2,2)
        self.conv4_4 = Conv2D(30,(3,3), strides=(1,1), activation=tf.nn.leaky_relu, padding='same')
        self.conv4_5 = Conv2D(30,(3,3), strides=(1,1), activation=tf.nn.leaky_relu, padding='same')
        self.conv4_6 = Conv2D(30,(3,3), strides=(1,1), activation=tf.nn.leaky_relu, padding='same')
        
        self.flat = Flatten()
        self.dense1 = Dense(128, activation=tf.nn.leaky_relu)
        self.dense2 = Dense(64, activation=tf.nn.leaky_relu)
        self.dense3 = Dense(10, activation = tf.nn.softmax)

        #NOTE --> this point is 230401's solution. same tool "real AND dummy CONV"
        self.dummy1_1 = Conv2D(1,(3,3), strides = (1,1), padding = 'same', trainable= False)
        self.dummy1_2 = Conv2D(1,(3,3), strides = (1,1), padding = 'same', trainable= False)
        self.dummy2_1 = Conv2D(1,(3,3), strides = (1,1), padding = 'same', trainable= False)
        self.dummy2_2 = Conv2D(1,(3,3), strides = (1,1), padding = 'same', trainable= False)
        self.dummy3_1 = Conv2D(1,(3,3), strides = (1,1), padding = 'same', trainable= False)
        self.dummy3_2 = Conv2D(1,(3,3), strides = (1,1), padding = 'same', trainable= False)
        self.dummy3_3 = Conv2D(1,(3,3), strides = (1,1), padding = 'same', trainable= False)
        self.dummy4_1 = Conv2D(1,(3,3), strides = (1,1), padding = 'same', trainable= False)
        self.dummy4_2 = Conv2D(1,(3,3), strides = (1,1), padding = 'same', trainable= False)
        self.dummy4_3 = Conv2D(1,(3,3), strides = (1,1), padding = 'same', trainable= False)
        self.dummy4_4 = Conv2D(1,(3,3), strides = (1,1), padding = 'same', trainable= False)
        self.dummy4_5 = Conv2D(1,(3,3), strides = (1,1), padding = 'same', trainable= False)
        self.dummy4_6 = Conv2D(1,(3,3), strides = (1,1), padding = 'same', trainable= False)
        
    def call(self, x): 
        #TODO  --> many cases for Delgate Optimizing Test.
        #1. concate 4 times --> model_mnist_11.tflite
        #2. 230401 : but mnist_11.tflite is low accuracy (~0.19) 
        #2.          what's the matter ?? 
        #2.          Is that right : first dummy input {1,30,30,1}
        #2. 230401 : find solution : num of parameters
        #            VGG DNN is not suitable model for MNIST data
        model_dummy_1_1 = self.dummy1_1(tf.zeros([1,28,28,1]))
        model_dummy_1_1.trainable =  False
        model_dummy_1_2 = self.dummy1_1(tf.zeros([1,28,28,1]))
        model_dummy_1_2.trainable =  False
        net = self.conv1_1(x)
        net = concatenate([net, model_dummy_1_1], axis=3)
        net = self.conv1_2(net)
        net = concatenate([net,model_dummy_1_2], axis=3)
        net = self.pool1(net)

        net = self.conv2_1(net)
        model_dummy_2_1 = self.dummy2_1(tf.zeros([1,14,14,1]))
        model_dummy_2_1.trainable =  False
        net = concatenate([net, model_dummy_2_1], axis=3)
        net = self.conv2_2(net)
        model_dummy_2_2 = self.dummy2_2(tf.zeros([1,14,14,1]))
        model_dummy_2_2.trainable =  False
        net = concatenate([net,model_dummy_2_2],axis=3)
        net = self.pool2(net)
       
        model_dummy_3_1  = self.dummy3_1(tf.zeros([1,7,7,1]))
        model_dummy_3_1.trainable =  False
        net = self.conv3_1(net)
        net = concatenate([net,model_dummy_3_1], axis=3)

        model_dummy_3_2  = self.dummy3_2(tf.zeros([1,7,7,1]))
        model_dummy_3_2.trainable =  False
        net = self.conv3_2(net)
        net = concatenate([net,model_dummy_3_2], axis=3)

        net = self.conv3_3(net)
        model_dummy_3_3  = self.dummy3_3(tf.zeros([1,7,7,1]))
        model_dummy_3_3.trainable =  False
        net = concatenate([net,model_dummy_3_3], axis=3)
        net = self.pool3(net)


        net = self.conv4_1(net)
        model_dummy_4_1  = self.dummy4_1(tf.zeros([1,3,3,1]))
        model_dummy_4_1.trainable = False
        net = concatenate([net,model_dummy_4_1], axis=3)


        net = self.conv4_2(net)
        model_dummy_4_2  = self.dummy4_2(tf.zeros([1,3,3,1]))
        model_dummy_4_2.trainable = False
        net = concatenate([net,model_dummy_4_2], axis=3)

        net = self.conv4_3(net)
        model_dummy_4_3  = self.dummy4_3(tf.zeros([1,3,3,1]))
        model_dummy_4_3.trainable = False
        net = concatenate([net,model_dummy_4_3], axis=3)
        net = self.pool4(net)


        net = self.conv4_4(net)
        model_dummy_4_4  = self.dummy4_4(tf.zeros([1,1,1,1]))
        model_dummy_4_4.trainable = False
        net = concatenate([net,model_dummy_4_4], axis=3)

        net = self.conv4_5(net)
        model_dummy_4_5  = self.dummy4_5(tf.zeros([1,1,1,1]))
        model_dummy_4_5.trainable = False
        net = concatenate([net,model_dummy_4_5], axis=3)

        net = self.conv4_6(net)
        model_dummy_4_6  = self.dummy4_6(tf.zeros([1,1,1,1]))
        model_dummy_4_6.trainable = False
        net = concatenate([net,model_dummy_4_6], axis=3)

        net = self.flat(net)
        net = self.dense1(net)
        net = self.dense2(net)
        net = self.dense3(net)
        return net
    def model(self):
        x = Input(shape=input_shape)
        return Model(inputs=[x], outputs=self.call(x))

myGpuModel = model_VGG().model()
myGpuModel.build([1,28,28,1])
myGpuModel.summary()
#plot_model(myGpuModel,to_file = 'model_gpu.png',show_shapes=True)
myGpuModel.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#HOON : adam -> SGD
#HOON : 


myGpuModel.fit(x=x_train, y=y_train, batch_size=1, epochs= 5)

test_loss, test_acc = myGpuModel.evaluate(x_test, y_test, batch_size=1)

print('accuacy :', test_acc)

myGpuModel.save('saved_model/mnist_13')

#--------------------------------------------------------------------------------------
#NOTE just 3-times CONV MNIST Model by sequential class
#model = Sequential()
#model.add(Conv2D(10, kernel_size=(3,3), input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(20, kernel_size=(3,3)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(40, kernel_size=(3,3)))
#model.add(Flatten())
#model.add(Dense(128, activation=tf.nn.relu))
#model.add(Dropout(0.2))
#model.add(Dense(10,activation=tf.nn.softmax))
#model.summary()
#
#
#model.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])
#model.fit(x=x_train,y=y_train, epochs=10)
#
#test_loss, test_acc = model.evaluate(x_test, y_test)
#
#print('accuacy :', test_acc)
#
#model.save('saved_model/new_cpu')
#
#--------------------------------------------------------------------------------------

