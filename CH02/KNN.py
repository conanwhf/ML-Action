#!/usr/bin/python3
#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt

'''
k近邻算法-对未知类别属性的数据集中的每个点依次执行以下操作：
 1. 计算已知类别数据集中的点与当前点之间的距离；
 2. 按照距离递增次序排序；
 3. 选取与当前点距离最小的k个点；
 4. 确定前k个点所在类别的出现频率；
 5. 返回前k个点出现频率最高的类别作为当前点的预测分类。
'''

class KNN(object):
	def __init__(self, labels=[]):#特征点的属性名称
		self.dataSet = pd.DataFrame(columns=labels+ ['Class'])
		self.count = 0
		return
		
	def __exit__(self):
		self.count = 0
		self.dataSet = pd.DataFrame()
		return
		 
	# 获取、设置类中的dataSet，可以直接通过this.dataSet操作
	def getDataSet(self):
		return (self.dataSet)
		
	def setDataSet(self, data):
		self.dataSet = data
		self.count = len(data)
		return 	
	
	# 添加特征值矩阵的数据
	def addDataSet(self, data):
		new = pd.DataFrame(data)
		self.dataSet = self.dataSet.append(new)
		self.dataSet = self.dataSet.reset_index(drop=True)
		self.count = len(self.dataSet)
		return
		
	# 从文件中读入特征矩阵	
	def getDataSetByFile(self, fn, names, sep="	"):
		self.dataSet = pd.read_table(fn,sep=sep,names=names,header=None,engine='python')
		self.count = len(self.dataSet)
		return self.dataSet
		
	# 分类器，可以传合格的dataframe，或者留空来使用类内部的dataSet
	def classify(self, input, k, data=pd.DataFrame()):
		if data.empty:
			data = self.dataSet
		#计算欧式距离
		# 1. 生成重复矩阵，每个元素都是待归类数据
		new = pd.DataFrame([input], index=range(0, len(data)))
		# 2. 生成diff矩阵
		diffMat = new - data
		#print("diffMat\n", diffMat)
		# 3. 对diff矩阵中每个数据进行计算，得出欧式距离
		sqDiffMat = diffMat**2
		sqDistances = sqDiffMat.sum(axis=1)
		data['Distances'] = sqDistances**0.5
		# 4. 对计算得到的距离排序
		data = data.sort_values(by=['Distances'])
		#print("New distances mat:\n",self.dataSet)
		# 5. 选择距离最小的k个点， 统计Class中各个值出现的次数
		res = data.head(k)['Class'].value_counts()
		return res.index[0]
	

'''绘制散点图'''
def draw(x, y, color='b'):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(x, y, c=color)
	plt.show()
	return


''' Main
'''
if __name__ == "__main__":
	''' Part1，简单测试KNN算法
	knn = KNN(['P1', 'P2'])	
	knn.addDataSet({'P1':[1.0, 1.1, 0.1, 0.2], 
						'P2':[1.1, 1.0, 2.0, 2.1],
						'Class': ['A', 'A', 'B', 'B']})
	knn.addDataSet({'P1':[1.0, 3.1, 0.2, 0.3], 
							'P2':[1.2, 0.0, 7.0, 4.1],
							'Class': ['A', 'A', 'B', 'B']})					
	print(knn.classify({'P1':1.1, 'P2':1.0}, 5))
	del(knn)
	'''

	''' Part2，计算'''
	dating = KNN()
	dating.getDataSetByFile(fn= 'datingTestSet.txt',
				sep='[\s,\t,\,]+', 
				names=['Flying','TVgame','IceCream','Class'])
	#根据分类生成数字指代的分类表，并绘制图像
	classList = dating.dataSet['Class'].replace({"largeDoses":2, "smallDoses":1, "didntLike":0})
	draw(dating.dataSet['IceCream'], dating.dataSet['Flying'],
			color=30*classList)
	
	
	