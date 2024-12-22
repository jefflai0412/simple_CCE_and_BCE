import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def MY_L1(yHat, y):
    return 0.5 * K.mean(K.square(y - yHat))


# 載入MNIST資料集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data( )

# 建立人工神經網路
network = Sequential( )
network.add( Dense( 512, activation ='relu', input_shape = ( 784, ) ) )
network.add( Dense( 10, activation = 'softmax' ) )
network.compile( optimizer = 'rmsprop', loss = MY_L1, metrics = ['accuracy'] )
print( network.summary() )
# 資料前處理
train_images = train_images.reshape( ( 60000, 28 * 28 ) )
train_images = train_images.astype( 'float32' ) / 255
test_images = test_images.reshape( ( 10000, 28 * 28 ) )
test_images = test_images.astype( 'float32' ) / 255
train_labels = to_categorical( train_labels )
test_labels = to_categorical( test_labels )

# 訓練階段
network.fit( train_images, train_labels, epochs = 5, batch_size = 200 )
# 測試階段
test_loss, test_acc = network.evaluate( test_images, test_labels )
print( "Test Accuracy:", test_acc )

