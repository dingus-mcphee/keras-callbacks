from tensorflow.python.keras.callbacks import Callback

from datetime import timedelta
from datetime.datetime import now

class timer(Callback):
	'''
		This class makes sure that training does not
	take more time than you want to in a single training
	session.
	'''

	def __init__(self, max_time):
		'''
		This init method takes a max time in seconds.
		'''

		super(Callback, self).__init__()

		self.max_time = timedelta(0, max_time)

		self.t0 = 0
		self.t1 = 0

	def on_epoch_begin(self, epoch, logs):
		if(epoch == 0):
			self.t0 = now()

	def on_epoch_end(self, epoch, logs):
		self.t1 = now()
		if((self.t1-self.t0) > self.max_time):
			self.model.stop_training = True
