import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalization
x_train = x_train/255 # x_norm = (x-xmin)/(xmax - xmin) = (x-0)/(255-0) = x/255
x_test = x_test/255
x_train.shape


# Set input shape
sample_shape = x_train[0].shape
img_width, img_height = sample_shape[0], sample_shape[1]
input_shape = (img_width, img_height, 1) #1 for grayscale

# Reshape data 
x_train = x_train.reshape(len(x_train), input_shape[0], input_shape[1], input_shape[2])
x_test  = x_test.reshape(len(x_test), input_shape[0], input_shape[1], input_shape[2])

# NN Architecture
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32,(3,3),input_shape=input_shape)) 
model.add(tf.keras.layers.MaxPooling2D()) 
model.add(tf.keras.layers.Flatten(input_shape = input_shape))
model.add(tf.keras.layers.Dense(392,input_shape = input_shape,activation = 'relu')) 
model.add(tf.keras.layers.Dense(10, activation= 'softmax'))

# compiling the model
model.compile(loss='SparseCategoricalCrossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10)


model.save('model'))
