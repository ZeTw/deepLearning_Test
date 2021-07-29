import numpy as np
import matplotlib.pyplot as plt

# data_train = open("dataSet/mnist_train_100.csv", 'r')
# data_list = data_train.readlines()
# data_train.close()
#
# all_value = data_list[0].split(',')
# print(type(all_value[0]))
# # image_array = np.asarray(all_value[1:], 'float64')
# # print(image_array)
# # print(type(image_array[0]))
# image_array = np.asfarray(all_value[1:]).reshape((28, 28))  # asfarray() 转换为一个浮点型的数组
# scaled_input = (image_array / 255.0 * 0.99) + 0.01
# print(scaled_input)
# plt.imshow(image_array, cmap='Greys', interpolation='None')
#
# plt.show()
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
        数据集处理：由于使用sigmoid函数，将目标值设置到 0.01--1的范围内 并设置目标值集合
        '''

    def show_num(self, image_array):
        image_array = image_array.reshape((28, 28))
        plt.imshow(image_array, cmap='Greys', interpolation='None')
        plt.show()

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

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


neural = neuralNetwork(784, 200, 10, 0.1)
target, train, num = neural.data_deal("dataSet/mnist_train.csv")
# print(num)
# for record in range(5):
score = []
for i, j, n in zip(train, target, num):
    # print(i.shape)
    # print(j.shape)
    neural.train(i, j)
    outputs = neural.query(i)
    # print(np.argmax(outputs))
    if n == np.argmax(outputs):
        score.append(n)
print(float(len(score) / len(num)))
test_target, test_train, test_num = neural.data_deal("dataSet/mnist_test_10.csv")
test_score = []
# for i in test_train:
#     neural.show_num(i)
for o, p, q in zip(test_train, test_target, test_num):
    # print(i.shape)
    # print(j.shape)
    outputs = neural.query(o)
    print(np.argmax(outputs))
    if q == np.argmax(outputs):
        test_score.append(q)

print(float(len(test_score) / len(test_num)))
