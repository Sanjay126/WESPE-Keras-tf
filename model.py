
from keras.models import Model
from kers.activations import tanh,relu,softmax
from keras.layers.advanced_activations import LeakyReLU
import keras
from keras import backend as K
class ConvBlock(keras.Model):
	def __init__(self):
		super(ConvBlock,self).__init__()
		self.conv1 = keras.layers.Conv2d(64, (3,3), padding=1)
		self.conv2 = keras.layers.Conv2d(64, (3,3), padding=1)
		self.bn1 = keras.layers.BatchNormalization(axis=-1)
		self.bn2 = keras.layers.BatchNormalization(axis=-1)

	def call(self,inputs):
		y = relu(self.bn1(self.conv1(inputs)))
		y = relu(self.bn2(self.conv2(y)))+ inputs
		return y

class Generator(keras.Model):

	def __init__(self,input_shape):
		super(Generator, self).__init__()
		self.conv1 = keras.layers.Conv2d(64, (9,9),input_shape=input_shape ,padding=4)
		self.block1 = ConvBlock()
		self.block2=ConvBlock()
		self.block3=ConvBlock()
		self.block4=ConvBlock()
		self.conv2 = keras.layers.Conv2d(64, (3,3), padding=1)
		self.conv3 = keras.layers.Conv2d(64, (3,3), padding=1)
		self.conv4 = keras.layers.Conv2d(64, (9,9), padding=4)
	def call(self,inputs):
		y=relu(self.conv1(inputs))
		y=self.block4(self.block3(self.block2(self.block1(y))))
		return tanh(self.conv4(relu(self.conv3(relu(self.conv2(y))))))

class Discriminator(keras.Model):
	def __init__(self,input_shape):
		super(Discriminator,self).__init__()
		self.conv1=keras.layers.Conv2d(48, (11, 11), input_shape=input_shape,stride=4, padding=5)
		self.relu1=LeakyReLU(negative_slope=0.2, inplace=True)
		self.conv2=keras.layers.Conv2d(128, (5,5), stride=2, padding=2)
		self.bn1=keras.layers.BatchNormalization(axis=-1)
		self.relu2=LeakyReLU(negative_slope=0.2, inplace=True)
		self.conv3=keras.layers.Conv2d(192, (3,3), stride=1, padding=1)
		self.bn2=keras.layers.BatchNormalization(axis=-1)
		self.relu3=LeakyReLU(negative_slope=0.2, inplace=True)
		self.conv4=keras.layers.Conv2d(192, (3,3), stride=1, padding=1)
		self.bn3=keras.layers.BatchNormalization(axis=-1)
		self.relu4=LeakyReLU(negative_slope=0.2, inplace=True)
		
		self.conv5=keras.layers.Conv2d(128, (3,3), stride=2, padding=1)
		self.bn4=keras.layers.BatchNormalization(axis=-1)
		self.relu5=LeakyReLU(negative_slope=0.2, inplace=True)
		
		self.fc = keras.layers.Dense(128*7*7, 1024)
		self.relu6=LeakyReLU(negative_slope=0.2)
		self.out = keras.layers.Dense(1024, 2) 

	def call(self, inputs):
		y = self.relu1(self.conv1(inputs))
		y = self.relu2(self.bn1(self.conv2(y)))
		y = self.relu3(self.bn2(self.conv3(y)))
		y = self.relu4(self.bn3(self.conv4(y)))
		y = self.relu5(self.bn4(self.conv5(y)))
		y=self.relu6(self.fc(y))
		return softmax(self.out(y))



