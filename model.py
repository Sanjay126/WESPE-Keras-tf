import tensorflow as tf
from keras.models import Model
from keras.activations import tanh,relu,softmax
from keras.layers.advanced_activations import PReLU
import keras
from keras import backend as K
from keras.applications import MobileNetV2
import numpy as np
from keras.layers import DepthwiseConv2D
from keras.applications.mobilenet_v2 import MobileNetV2
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

class GaussianBlur(keras.layers.Layer):
	def __init__(self):
		super(GaussianBlur,self).__init__()

		kernel_size = 3  # set the filter size of Gaussian filter
		kernel_weights = [[0.03797616, 0.044863533, 0.03797616],
                  [0.044863533, 0.053, 0.044863533],
                  [0.03797616, 0.044863533, 0.03797616]]

		in_channels = 3 
		kernel_weights = np.expand_dims(kernel_weights, axis=-1)
		kernel_weights = np.repeat(kernel_weights, in_channels, axis=-1)
		kernel_weights = np.expand_dims(kernel_weights, axis=-1)
		
		self.g_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
		self.g_layer.set_weights([kernel_weights])
		g_layer.trainable = False 
		

	def call(self,inputs):
		g_layer_out = self.g_layer(inputs)  
		return g_layer_out


class GrayScale(keras.layers.Layer):
	def __init__(self):
		super(GrayScale,self).__init__()

	def call(self,image):
		r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

		return gray

class Generator(keras.layers.Layer):

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


class Discriminator(keras.layers.Layer):
	def __init__(self,input_shape):
		super(Discriminator,self).__init__()
		self.conv1=keras.layers.Conv2d(48, (11, 11), input_shape=input_shape,stride=4, padding=5)
		self.relu1=PReLU()
		self.conv2=keras.layers.Conv2d(128, (5,5), stride=2, padding=2)
		self.bn1=keras.layers.BatchNormalization(axis=-1)
		self.relu2=PReLU()
		self.conv3=keras.layers.Conv2d(192, (3,3), stride=1, padding=1)
		self.bn2=keras.layers.BatchNormalization(axis=-1)
		self.relu3=PReLU()
		self.conv4=keras.layers.Conv2d(192, (3,3), stride=1, padding=1)
		self.bn3=keras.layers.BatchNormalization(axis=-1)
		self.relu4=PReLU()
		
		self.conv5=keras.layers.Conv2d(128, (3,3), stride=2, padding=1)
		self.bn4=keras.layers.BatchNormalization(axis=-1)
		self.relu5=PReLU()
		
		self.fc = keras.layers.Dense(128*7*7, 1024)
		self.relu6=PReLU()
		self.out = keras.layers.Dense(1024, 2) 


	def call(self, inputs):
		y = self.relu1(self.conv1(inputs))
		y = self.relu2(self.bn1(self.conv2(y)))
		y = self.relu3(self.bn2(self.conv3(y)))
		y = self.relu4(self.bn3(self.conv4(y)))
		y = self.relu5(self.bn4(self.conv5(y)))
		y=self.relu6(self.fc(y))
		return softmax(self.out(y))



class WESPE:
	def __init__(self,gen_input_shape,disc_input_shape):
		self.generator_g=Generator(gen_input_shape)
		#TODO: add compile arguments
		# self.generator_g.compile()
		
		self.generator_f=Generator(disc_input_shape)
		# self.generator_f.compile()

		self.discriminator_c=Discriminator(disc_input_shape)
		# self.discriminator_c.compile()

		self.discriminator_t=Discriminator(disc_input_shape)
		# self.discriminator_t.compile()

		self.blur=GaussianBlur()
		self.blur.trainable=False

		self.gray=GrayScale()
		self.gray.trainable=False
		self.mobilenet=MobileNetV2(gen_input_shape)

	def train_step(self,x,y):

		batch_size=x.shape[0]
		with tf.GradientTape() as tape:
			self.discriminator_c.trainable=False
			self.discriminator_t.trainable=False

			y_fake=self.generator_g(x)
			x_fake=self.generator_f(y_fake)

			mobilenet_x_true=self.mobilenet.predict(x)
			mobilenet_x_fake=self.mobilenet.predict(x_fake)


			y_real_blur=self.blur(y)
			y_fake_blur=self.blur(y_fake)

			y_fake_blur_pred=self.discriminator_c(y_fake_blur)
			y_real_blur_pred=self.discriminator_c(y_real_blur)

			y_fake_gray=self.gray(y_fake)
			y_real_gray=self.gray(y)

			y_fake_gray_pred=self.discriminator_t(y_fake_gray)
			y_real_gray_pred=self.discriminator_t(y_real_gray)

			content_loss=self.content_loss(mobilenet_x_fake,mobilenet_x_true)
			tv_loss=self.tv_loss(y_fake)
			dc_loss_g=self.color_loss(y_fake_blur_pred,tf.ones((batch_size,1)))
			dt_loss_g=self.texture_loss(y_fake_gray_pred,tf.ones((batch_size,1)))

			net_loss=content_loss+10*tv_loss+ 0.005*(dc_loss_g+dt_loss_g)

			grads = tape.gradient(net_loss, self.generator_g.trainable_weights)
			gen_g_optimizer.apply_gradients(zip(grads, self.generator_g.trainable_weights))
			grads = tape.gradient(net_loss, self.generator_f.trainable_weights)
			gen_f_optimizer.apply_gradients(zip(grads, self.generator_f.trainable_weights))


		with tf.GradientTape() as tape:

			y_fake_blur_pred=self.discriminator_c(y_fake_blur)
			dc_loss=self.color_loss(y_fake_blur_pred,tf.ones((batch_size,1)))+self.color_loss(y_real_blur_pred,tf.zeros((batch_size,1)))
			grads=tape.gradient(dc_loss,self.discriminator_c.trainable_weights)
			disc_c_optimizer.apply_gradients(zip(grads),self.discriminator_c.trainable_weights)
			
			y_fake_gray_pred=self.discriminator_t(y_fake_gray)
			dt_loss=self.texture_loss(y_fake_gray_pred,tf.ones((batch_size,1)))+self.texture_loss(y_real_gray_pred,tf.zeros((batch_size,1)))
			grads=tape.gradient(dt_loss,self.discriminator_t.trainable_weights)
			disc_c_optimizer.apply_gradients(zip(grads),self.discriminator_t.trainable_weights)




