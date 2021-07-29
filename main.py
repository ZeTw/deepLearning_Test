import numpy as np
import matplotlib.pyplot as plt
import scipy.special


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class neuralNetwork:
    """
    初始化
    """

    def __init__(self, innodes, hinodes, outnodes, learn):
        self.innodes = innodes
        self.hinodes = hinodes
        self.outnodes = outnodes
        self.wih = np.random.normal(0.0, pow(self.innodes, -0.5), (self.hinodes, self.innodes))  # (100,784)
        self.who = np.random.normal(0.0, pow(self.hinodes, -0.5), (self.outnodes, self.hinodes))  # (10,100)
        self.lr = learn
        self.activation_function = lambda x: scipy.special.expit(x)
   '''
    以图片形式展示数据集中的数字
   '''
      
    def show_num(self, image_array):
        image_array = image_array.reshape((28, 28))
        plt.imshow(image_array, cmap='Greys', interpolation='None')
        plt.show()
        
    '''
    数据集处理：由于使用sigmoid函数，将目标值设置到 0.01--1的范围内 并设置目标值集合
    '''
    def data_deal(self, data_path):
        data_train = open(data_path, 'r')
        data_list = data_train.readlines()
        data_train.close()
        train_list = []
        target_list = []
        num_list = []
        out_nodes = 10
        for value in data_list:
            train_value = value.split(',')
            train_list.append(np.asfarray(train_value[1:]) / 255.0 * 0.99 + 0.01)
            num_list.append(np.asfarray(train_value[0]))
            targets = np.zeros(out_nodes) + 0.01
            targets[int(train_value[0])] = 0.99
            target_list.append(targets)
        return np.array(target_list), np.array(train_list), np.array(num_list)
    '''
    训练函数：前向传播计算输出，同时反向传播更新权重矩阵
    使用的激活函数为sigmoid
    '''
    def train(self, train_set, target_set):
        input = np.array(train_set, ndmin=2).T
        target = np.array(target_set, ndmin=2).T

        hidden_input = np.dot(self.wih, input)
        hidden_output = self.activation_function(hidden_input)
        final_input = np.dot(self.who, hidden_output)
        final_output = self.activation_function(final_input)

        output_error = target - final_output
        hidden_error = np.dot(self.who.T, output_error)

        self.who += self.lr * np.dot((output_error * final_output * (1.0 - final_output)), np.transpose(hidden_output))
        self.wih += self.lr * np.dot((hidden_error * hidden_output * (1.0 - hidden_output)), np.transpose(input))
     
      '''
      使用train得到的权重矩阵计算最终输出值，用以和真实值进行对比，从而计算该神经网络的正确率
      '''
    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# 初始化
neural = neuralNetwork(784, 200, 10, 0.1)

'''
# 一开始使用100个数据的数据集进行训练，训练5次，虽然训练时的准确率挺高的 99%左右，但是在使用测试集进行测试时，准确率降到了60%
#推测为训练数据集中的数据不够随机，导致测试时准确度下降
target, train, num = neural.data_deal("dataSet/mnist_train_100.csv")
for record in range(5):
    score = []
    for i, j, n in zip(train, target, num):
        neural.train(i, j)
        outputs = neural.query(i)
        if n == np.argmax(outputs):
            score.append(n)
print(float(len(score) / len(num)))
'''
# 改用包含10000个数据的数据集进行训练，准确度提升至98% 且测试时的准确度在97%左右
target, train, num = neural.data_deal("dataSet/mnist_train.csv")
score = []
for i, j, n in zip(train, target, num):
    neural.train(i, j)
    outputs = neural.query(i)
   
    if n == np.argmax(outputs):
        score.append(n)
print(float(len(score) / len(num)))

# 由于测试集数据比较少，因此改用含有100个数据的训练集进行测试
test_target, test_train, test_num = neural.data_deal("dataSet/mnist_train_100.csv")
test_score = []

for o, p, q in zip(test_train, test_target, test_num):
    outputs = neural.query(o)
    print(np.argmax(outputs))
    if q == np.argmax(outputs):
        test_score.append(q)

print(float(len(test_score) / len(test_num)))
