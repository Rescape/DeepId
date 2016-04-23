import mxnet as mx

def get_symbol(num_class = 100):
	input_data = mx.symbol.Variable(name='data')

	# group 1
	conv1 = mx.symbol.Convolution(
		data=input_data, kernel=(4, 4), stride=(1, 1), num_filter=20, name="conv1")
	# batch norm
	bn1 = mx.symbol.BatchNorm(data=conv1, eps=0.001, momentum=0.9, fix_gamma=True, name="bn1")
	relu1 = mx.symbol.Activation(data=bn1, act_type="relu", name="relu1")
	pool1 = mx.symbol.Pooling(data=relu1, pool_max="max", kernel=(2, 2), stride=(2, 2), name="pool1")

	# group 2
	conv2 = mx.symbol.Convolution(
		data=pool1, kernel=(3, 3), stride=(1, 1), num_filter=40, name="conv2")
	# batch norm
	bn2 = mx.symbol.BatchNorm(data=conv2, eps=0.001, momentum=0.9, fix_gamma=True, name="bn2")
	relu2 = mx.symbol.Activation(data=bn2, act_type="relu", name="relu2")
	pool2 = mx.symbol.Pooling(data=relu2, pool_max="max", kernel=(2, 2), stride=(2, 2), name="pool2")

	# group3
	conv3 = mx.symbol.Convolution(
		data=pool2, kernel=(3, 3), stride=(1, 1), num_filter=60, name="conv3")
	# batch norm
	bn3 = mx.symbol.BatchNorm(data=conv3, eps=0.001, momentum=0.9, fix_gamma=True, name="bn3")
	relu3 = mx.symbol.Activation(data=bn3, act_type="relu", name="relu3")
	pool3 = mx.symbol.Pooling(data=relu3, pool_max="max", kernel=(2, 2), stride=(2, 2), name="pool3")

	# group4
	conv4 = mx.symbol.Convolution(
		data=conv3, kernel=(2, 2), stride=(2, 2), num_filter=80, name="conv4")
	# batch norm
	bn4 = mx.symbol.BatchNorm(data=conv4, eps=0.001, momentum=0.9, fix_gamma=True, name="bn4")
	relu4 = mx.symbol.Activation(data=bn4, act_type="relu", name="relu4")
	pool4 = mx.symbol.Pooling(data=relu4, pool_max="max", kernel=(2, 2), stride=(2, 2), name="pool4")

	