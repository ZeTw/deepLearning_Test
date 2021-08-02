## 使用单隐层神经网络实现MNIST手写数字数据集的识别  
代码的训练集以及测试集的准确度如下图所示：  
训练集准确率：  
![训练集准确率](https://github.com/ZeTw/deepLearning_Test/blob/main/MNIST_Identify/%E8%AE%AD%E7%BB%83%E9%9B%86%E5%87%86%E7%A1%AE%E7%8E%87.png)  
测试集准确率：  
![测试集准确率](https://github.com/ZeTw/deepLearning_Test/blob/main/MNIST_Identify/%E6%B5%8B%E8%AF%95%E9%9B%86%E5%87%86%E7%A1%AE%E7%8E%87.png)  
main.py 代码并没有使用向量化的思想，训练以及测试集的检测均采用了for循环，我们知道向量化会使程序在速度上得到很大的提升，为此在原有代码的基础上进行改善，提高程序运行速度。
