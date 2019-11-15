
import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, MaxPooling2D, Convolution1D, Convolution2D, Flatten, Dense, Dropout, Reshape, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from time import time

__feature_shape__ = {
	'image': [300,300,3],	
	'vein': [300,300],	
	'xyprojection': [60,],
	'color': [36,],
	'texture': [18,],
	'fourier': [17,],
	'shape': [33,],
}

def get_conv_encoder(image, l2_rate=0.0):
	"""return convolutional neural network encoder with image as input"""

	image = Lambda(lambda img: img/255)(image)

	### conv1
	image = Convolution2D(filters=16,
							kernel_size=[3,3],
							padding='valid',
							activation=tf.nn.relu)(image)

	image = BatchNormalization(momentum=0.01)(image)
	image = MaxPooling2D(pool_size=[2,2],
							strides=[2,2],
							padding='same')(image)

	### conv2
	image = Convolution2D(filters=16,
							kernel_size=[3,3],
							padding='valid',
							activation=tf.nn.relu)(image)
	image = BatchNormalization(momentum=0.01)(image)
	image = MaxPooling2D(pool_size=[2,2],
							strides=[2,2],
							padding='same')(image)
	### conv3
	image = Convolution2D(filters=32,
							kernel_size=[5,5],
							padding='valid',
							activation=tf.nn.relu)(image)
	image = BatchNormalization(momentum=0.01)(image)
	image = MaxPooling2D(pool_size=[2,2],
							strides=[2,2],
							padding='same')(image)
	### conv4
	image = Convolution2D(filters=32,
							kernel_size=[5,5],
							padding='valid',
							activation=tf.nn.relu)(image)
	image = BatchNormalization(momentum=0.01)(image)
	image = MaxPooling2D(pool_size=[2,2],
							strides=[2,2],
							padding='same')(image)
	### conv5
	image = Convolution2D(filters=32,
							kernel_size=[5,5],
							padding='valid',
							activation=tf.nn.relu)(image)
	image = BatchNormalization(momentum=0.01)(image)
	image = MaxPooling2D(pool_size=[2,2],
							strides=[2,2],
							padding='same')(image)
	### flatten
	image = Flatten()(image)

	image = Dense(units=100, 
					activation=tf.nn.relu,
					kernel_regularizer=l2(l2_rate))(image)

	return image

def get_1dconv_encoder(proj, l2_rate=0.0):
	x = Reshape(target_shape=(60,1))(proj)

	x = Convolution1D(filters=16,
						kernel_size=3,
						strides=1,
						padding='valid',
						activation='relu')(x)
	x = BatchNormalization()(x)
	x = MaxPool1D(pool_size=2,
					padding='same')(x)

	x = Convolution1D(filters=16,
						kernel_size=3,
						strides=1,
						padding='valid',
						activation='relu')(x)
	x = BatchNormalization()(x)
	x = MaxPool1D(pool_size=2,
					padding='same')(x)

	x = Convolution1D(filters=32,
						kernel_size=5,
						strides=1,
						padding='valid',
						activation='relu')(x)
	x = BatchNormalization()(x)
	x = MaxPool1D(pool_size=2,
					padding='same')(x)

	x = Flatten()(x)
	x = Dense(units=100, 
				activation=tf.nn.relu, 
				kernel_regularizer=l2(l2_rate))(x)
	return x

def get_dense_encoder(vector, l2_rate=0.0):
	x = Dense(units=100, 
			  activation=tf.nn.relu,
			  kernel_regularizer=l2(l2_rate))(vector)
	return x


def get_training_model(feature='combine', l2_rate=0.0, dropout=0.5):
	if feature in __feature_shape__.keys():
		inputs = Input(shape=__feature_shape__[feature])
	else:
		raise Exception("Invalid feature name ...")

	if feature in ['vein', 'image']:
		if feature =='vein':
			reshape = Reshape([300,300,1])(inputs)
		else: reshape = inputs
		encoder = get_conv_encoder(reshape, l2_rate=l2_rate)
	elif feature in ['xyprojection']:
		encoder = get_1dconv_encoder(inputs, l2_rate=l2_rate)
	else:
		encoder = get_dense_encoder(inputs, l2_rate=l2_rate)

	## add a softmax layer
	dropout = Dropout(rate=dropout)(encoder)
	softmax = Dense(units=32, activation='softmax')(dropout)

	model = Model(inputs, softmax)
	model.compile(optimizer='adam',
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])

	return model

class CheckpointCallback(tf.keras.callbacks.Callback):
	def __init__(self, early_stopping=1000, verbose=True):
		self.verbose = verbose
		self.best_val_loss = 1000.0
		self.n_no_improvements = 0
		self.training_time = time()
		self.early_stopping = early_stopping

	def on_epoch_end(self, epoch, logs):
		val_loss = logs['val_loss']
		if self.verbose and epoch % 1 == 0:
			print("Epoch {} - loss {} - acc {} - val_loss {} - val_acc {}".format(epoch, logs['loss'], logs['acc'], logs['val_loss'], logs['val_acc']))
		if val_loss < self.best_val_loss:
			self.n_no_improvements = 0
			self.best_val_loss = val_loss 
			self.best_weights = self.model.get_weights()

		else:
			self.n_no_improvements += 1
			if self.n_no_improvements > self.early_stopping:
				self.model.stop_training = True
				if self.verbose:
					print("Early stopping at epoch {}".format(epoch))
                
	def on_train_end(self, logs=None):
		if self.verbose:
			print('Restoring best model weights .')
		self.training_time = time() - self.training_time
		self.model.set_weights(self.best_weights)

class EncoderExtractor:

	def __init__(self):
		inputs = [Input(shape=input_shape) for input_shape in __feature_shape__.values()]

		#image
		image = inputs[0]
		image = get_conv_encoder(image)
		# vein
		vein = inputs[1]
		vein = get_conv_encoder(vein)
		# xyprojection
		xyproj = inputs[2]
		xyproj = get_1dconv_encoder(xyproj)
		# others
		handcrafted_features = [get_dense_encoder(vector) for vector in inputs[3:]]

		encoder_outputs_list = [image, vein, xyproj] + handcrafted_features
		combine = Concatenate()(encoder_outputs_list)
		self.extractor = Model(inputs, combine)

		self.encoders = []
		for input_tensor, representation in zip(inputs, encoder_outputs_list):
			softmax = Dense(units=32)(input_tensor)
			model = Model(input_tensor, softmax)
			model.compile(optimizer='adam',
						  loss='sparse_categorical_crossentropy',
						  metrics=['accuracy'])
			self.encoders.append(model)

	def extract(self, X):
		return self.extractor.predict(X)

	def load_encoders(self, modelpaths):
		for model, path in zip(self.encoders, modelpaths):
			print("Loading encoder ", path)
			model.load_weights(path)

