

import tensorflow as tf
import DataSets as ds
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LoadModel = False

experiment_size = 10
train = ds.DataSet('../DataBases/data_%dk.bin'%experiment_size,'../DataBases/gender_%dk.bin'%experiment_size,1000*experiment_size)
test = ds.DataSet('../DataBases/data_test10k.bin','../DataBases/gender_test10k.bin',10000)

class fc_layer(tf.Module):
	def __init__(self, name, input_dim, output_dim):
		self.scope_name = name
		w_init = tf.random.truncated_normal([input_dim, output_dim], stddev=0.1)
		self.w = tf.Variable(w_init)
		print('w      ', self.w.shape)
		b_init = tf.constant(0.0, shape=[output_dim])
		self.b = tf.Variable(b_init)
		print('b      ', self.b.get_shape())


	def __call__(self, x, log_summary):
		if log_summary:
			with tf.name_scope(self.scope_name) as scope:
				tf.summary.scalar("mean w", tf.reduce_mean(self.w))
				tf.summary.scalar("max w", tf.reduce_max(self.w))
				tf.summary.histogram("w", self.w)
				tf.summary.scalar("mean b", tf.reduce_mean(self.b))
				tf.summary.scalar("max b", tf.reduce_max(self.b))
				tf.summary.histogram("b", self.b)
		return tf.matmul(x, self.w) + self.b

class SimpleNet(tf.Module):
	def __init__(self, input_dim):
		self.fc1 = fc_layer('fc1', input_dim, 50)
		self.fc2 = fc_layer('fc2', 50, 2)

	def __call__(self, x, log_summary):
		x = self.fc1(x, log_summary)
		x = tf.nn.sigmoid(x)
		x = self.fc2(x, log_summary)
		return x

def train_one_iter(model, optimizer, image, label, log_summary):
	with tf.GradientTape() as tape:
		y = model(image,log_summary)
		loss = tf.reduce_sum(tf.square(y - label))
		if log_summary:
			tf.summary.scalar('loss', loss)
		grads = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))
	return loss


print ("-----------------------------------------------------")
print ("----------------------- %dk -------------------------"%experiment_size)
print ("-----------------------------------------------------")



train_summary_writer = tf.summary.create_file_writer('logs')
optimizer = tf.optimizers.SGD(1e-4)
simple_v2 = SimpleNet(train.dim)

for iter in range(5000):
	tf.summary.experimental.set_step(iter)
	ima, lab = train.NextTrainingBatch()
	with train_summary_writer.as_default():
		loss = train_one_iter(simple_v2, optimizer, ima, lab, iter % 10 == 0)
	if iter % 100 == 0:
		print("iter= %6d - loss= %f" % (iter, loss))

	if iter % 500 == 0:
		with train_summary_writer.as_default():
			acc1 = train.mean_accuracy(simple_v2)*100
			acc2 = test.mean_accuracy(simple_v2)*100
			print("iter= %6d accuracy - train= %.2f%% - test= %.2f%%" % (iter, acc1,acc2))



