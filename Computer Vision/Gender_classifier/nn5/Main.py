
import tensorflow as tf
import DataSets as ds
import Layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LoadModel = False

experiment_size = 10
train = ds.DataSet('../DataBases/data_%dk.bin'%experiment_size,'../DataBases/gender_%dk.bin'%experiment_size,1000*experiment_size)
test = ds.DataSet('../DataBases/data_test10k.bin','../DataBases/gender_test10k.bin',10000)

class ConvNeuralNet(tf.Module):
	def __init__(self):

		list = []
		list.append( Layers.unflat('unflat',48, 48, 1) )

		nbfilter = 3
		for i in range(4):
			for j in range(2):
				list.append(Layers.conv('block_%d_conv_%d'%(i,j), output_dim=nbfilter, filterSize=3, stride=1,dropout_rate = 0.1))
			list.append( Layers.maxpool('pool', 2) )
			nbfilter *= 2
		list.append(Layers.flat())
		list.append(Layers.fc('fc', 2))

		self.list = list

	def __call__(self, x, log_summary, training):
		for l in self.list:
			x = l(x, log_summary, training)
		return x


def train_one_iter(model, optimizer, image, label, log_summary):
	with tf.GradientTape() as tape:
		y = model(image,log_summary,True)
		y = tf.nn.log_softmax(y)
		diff = label * y
		loss = -tf.reduce_sum(diff)
		if log_summary:
			tf.summary.scalar('cross entropy', loss)
		grads = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))
	return loss

print ("-----------------------------------------------------")
print ("----------------------- %dk -------------------------"%experiment_size)
print ("-----------------------------------------------------")

train_summary_writer = tf.summary.create_file_writer('logs %dk'%experiment_size)
optimizer = tf.optimizers.Adam(1e-3)
simple_cnn = ConvNeuralNet()

for iter in range(5000):
	tf.summary.experimental.set_step(iter)

	if iter % 500 == 0:
		with train_summary_writer.as_default():
			acc1 = train.mean_accuracy(simple_cnn) * 100
			acc2 = test.mean_accuracy(simple_cnn) * 100
			print("iter= %6d accuracy - train= %.2f%% - test= %.2f%%" % (iter, acc1, acc2))

	ima, lab = train.NextTrainingBatch()
	with train_summary_writer.as_default():
		loss = train_one_iter(simple_cnn, optimizer, ima, lab, iter % 10 == 0)

	if iter % 100 == 0:
		print("iter= %6d - loss= %f" % (iter, loss))


exit()
import tensorflow as tf
import numpy as np
import DataSets as ds
import Layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_dict(database,IsTrainingMode):
	xs,ys = database.NextTrainingBatch()
	return {x:xs,y_desired:ys,ITM:IsTrainingMode}

LoadModel = False
KeepProb_Dropout = 0.9

experiment_name = '10k_Dr%.3f'%KeepProb_Dropout
#train = ds.DataSet('../DataBases/data_1k.bin','../DataBases/gender_1k.bin',1000)
train = ds.DataSet('../DataBases/data_10k.bin','../DataBases/gender_10k.bin',10000)
#train = ds.DataSet('../DataBases/data_100k.bin','../DataBases/gender_100k.bin',100000)
test = ds.DataSet('../DataBases/data_test10k.bin','../DataBases/gender_test10k.bin',10000)

with tf.name_scope('input'):
	x = tf.compat.v1.placeholder(tf.float32, [None, train.dim],name='x')
	y_desired = tf.compat.v1.placeholder(tf.float32, [None, 2],name='y_desired')
	ITM = tf.compat.v1.placeholder("bool", name='Is_Training_Mode')

with tf.name_scope('CNN'):
	t = Layers.unflat(x,48,48,1)
	nbfilter = 3
	for k in range(4):
		for i in range(2):
			t = Layers.conv(t,nbfilter,3,1,ITM,'conv_%d_%d'%(nbfilter,i),KeepProb_Dropout)
		t = Layers.maxpool(t,2,'pool')
		nbfilter *= 2
	
	t = Layers.flat(t)
	#t = Layers.fc(t,50,ITM,'fc_1',KeepProb_Dropout)
	y = Layers.fc(t,2,ITM,'fc_2',KP_dropout=1.0,act=tf.nn.log_softmax)

with tf.name_scope('cross_entropy'):
	diff = y_desired * y 
	with tf.name_scope('total'):
		cross_entropy = -tf.reduce_mean(diff)
	tf.compat.v1.summary.scalar('cross entropy', cross_entropy)
	
with tf.name_scope('accuracy'):
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_desired, 1))
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.compat.v1.summary.scalar('accuracy', accuracy)

with tf.name_scope('learning_rate'):
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.compat.v1.train.exponential_decay(1e-3,global_step,1000, 0.75, staircase=True)


with tf.name_scope('learning_rate'):
    tf.compat.v1.summary.scalar('learning_rate', learning_rate)

#train_step = tf.compat.v1.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)
train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)
merged = tf.compat.v1.summary.merge_all()

Acc_Train = tf.compat.v1.placeholder("float", name='Acc_Train');
Acc_Test = tf.compat.v1.placeholder("float", name='Acc_Test');
MeanAcc_summary = tf.compat.v1.summary.merge([tf.compat.v1.summary.scalar('Acc_Train', Acc_Train),tf.compat.v1.summary.scalar('Acc_Test', Acc_Test)])


print ("-----------------------------------------------------")
print ("-----------",experiment_name)
print ("-----------------------------------------------------")

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
writer = tf.compat.v1.summary.FileWriter(experiment_name, sess.graph)
saver = tf.compat.v1.train.Saver()
if LoadModel:
	saver.restore(sess, "./model.ckpt")

nbIt = 5000
for it in range(nbIt):
	trainDict = get_dict(train,IsTrainingMode=True)					
	sess.run(train_step, feed_dict=trainDict)
	
	if it%10 == 0:
		acc,ce,lr = sess.run([accuracy,cross_entropy,learning_rate], feed_dict=trainDict)
		print ("it= %6d - rate= %f - cross_entropy= %f - acc= %f" % (it,lr,ce,acc ))
		summary_merged = sess.run(merged, feed_dict=trainDict)
		writer.add_summary(summary_merged, it)	
				
	if it%100 == 50:
		Acc_Train_value = train.mean_accuracy(sess,accuracy,x,y_desired,ITM)
		Acc_Test_value = test.mean_accuracy(sess,accuracy,x,y_desired,ITM)
		print ("mean accuracy train = %f  test = %f" % (Acc_Train_value,Acc_Test_value ))
		summary_acc = sess.run(MeanAcc_summary, feed_dict={Acc_Train:Acc_Train_value,Acc_Test:Acc_Test_value})
		writer.add_summary(summary_acc, it)
		
writer.close()
if not LoadModel:
	saver.save(sess, "./model.ckpt")
sess.close()
