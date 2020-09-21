import tensorflow as tf
from tensorflow import keras

train_data = pd.read_csv('../input/mnist-in-csv/mnist_train.csv')
test_data = pd.read_csv('../input/mnist-in-csv/mnist_test.csv')

trainY = train_data['label']
trainY = tf.one_hot(trainY, 10)
trainX = train_data.drop(labels=['label'], axis=1)
trainX = trainX/255.0
trainX = trainX.values.reshape(-1, 28, 28, 1)

testY = test_data['label']
testY = tf.one_hot(testY, 10)
testX = test_data.drop(labels=['label'], axis=1)
testX = testX/255.0
testX = testX.values.reshape(-1, 28, 28, 1)

model1 = keras.Sequential([
    
    keras.layers.Conv2D(36,kernel_size=5, activation='relu',input_shape=(28,28,1)),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(64,kernel_size=5,activation='relu'),
    keras.layers.MaxPool2D((2,2)),
    
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10),
    keras.layers.Softmax()
])

model1.compile(optimizer=keras.optimizers.Adam(),
             loss=tf.keras.losses.CategoricalCrossentropy(),
             metrics=['accuracy'])

history = model1.fit(trainX, trainY, batch_size=128,
			 epochs=40, verbose=2)

test_loss, test_acc = model1.evaluate(testX, testY)
print(test_loss, test_acc)