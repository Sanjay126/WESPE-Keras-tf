
from keras.models import Model
from kers.activations import tanh,relu
import keras
from keras import backend as K
class ConvBlock(keras.layers.Layer):
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

class Generator(keras.layers.Layer):

	def __init__(self):
		super(Generator, self).__init__()
		self.conv1 = keras.layers.Conv2d(64, (9,9), padding=4)
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

class Discriminator(keras.layers.Layer):
	def__init(self):
	super(Discriminator,self).__init__()
	self.=keras.layers.Conv2d(input_ch, 48, 11, stride=4, padding=5)
    self.=keras.layers.LeakyReLU(negative_slope=0.2, inplace=True)
    self.=keras.layers.Conv2d(48, 128, 5, stride=2, padding=2)
    self.=keras.layers.LeakyReLU(negative_slope=0.2, inplace=True)
    self.=keras.layers.InstanceNorm2d(128, affine=True)
    self.=keras.layers.Conv2d(128, 192, 3, stride=1, padding=1)
    self.=keras.layers.LeakyReLU(negative_slope=0.2, inplace=True)
    self.=keras.layers.InstanceNorm2d(192, affine=True)
    self.=keras.layers.Conv2d(192, 192, 3, stride=1, padding=1)
    self.=keras.layers.LeakyReLU(negative_slope=0.2, inplace=True)
    self.=keras.layers.InstanceNorm2d(192, affine=True)
    self.=keras.layers.Conv2d(192, 128, 3, stride=2, padding=1)
    self.=keras.layers.LeakyReLU(negative_slope=0.2, inplace=True)
    self.=keras.layers.InstanceNorm2d(128, affine=True)
    self.fc = keras.layers.Dense(128*7*7, 1024)
    self.out = keras.layers.Dense(1024, 2) 


# class WESPE(keras.Model):
# 	def __init__(self, use_bn=False, use_dp=False, num_classes=10):
# 		super(WESPE, self).__init__(name='mlp')
# 		self.use_bn = use_bn
# 		self.use_dp = use_dp
# 		self.num_classes = num_classes

# 		self.dense1 = keras.layers.Dense(32, activation='relu')
# 		self.dense2 = keras.layers.Dense(num_classes, activation='softmax')
# 		if self.use_dp:
# 			self.dp = keras.layers.Dropout(0.5)
# 		if self.use_bn:
# 			self.bn = keras.layers.BatchNormalization(axis=-1)
# 	def call(self, inputs):
# 		x = self.dense1(inputs)
# 		if self.use_dp:
# 			x = self.dp(x)
# 		if self.use_bn:
# 			x = self.bn(x)
# 		return self.dense2(x)

