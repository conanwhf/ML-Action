#!/usr/bin/python3
#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import operator

'''
k近邻算法-对未知类别属性的数据集中的每个点依次执行以下操作：
 1. 计算已知类别数据集中的点与当前点之间的距离；
 2. 按照距离递增次序排序；
 3. 选取与当前点距离最小的k个点；
 4. 确定前k个点所在类别的出现频率；
 5. 返回前k个点出现频率最高的类别作为当前点的预测分类。
'''

class KNN(object):
	def __init__(self, labels):#数据点的属性个数
		
		self.dataSet = pd.DataFrame(columns=labels+ ['Class'])
		self.count = 0
		return

	def __exit__(self):
		pass

	def addDataSet(self, data):
		new = pd.DataFrame(data)
		self.dataSet = self.dataSet.append(new)
		self.dataSet = self.dataSet.reset_index(drop=True)
		self.count = len(self.dataSet)
		#print(self.dataSet)
		return
		
	def getDataSet(self):
		return (self.dataSet)
	
	def classify(self, data, k):
		dataSetSize = self.dataSet.shape[0]
		 #距离计算
		#print(np.tile(data, (dataSetSize,1)))
		new = pd.DataFrame([data], index=range(0, self.count))
		#print(new)
		diffMat = new - self.dataSet
		#print("diffMat\n", diffMat)
		sqDiffMat = diffMat**2
		sqDistances = sqDiffMat.sum(axis=1)
		self.dataSet['Distances'] = sqDistances**0.5
		self.dataSet = self.dataSet.sort_values(by=['Distances'])
		print("New distances mat:\n",self.dataSet)
		#选择距离最小的k个点， 统计Class中各个值出现的次数
		res = self.dataSet.head(k)['Class'].value_counts()
		return res.index[0]
		

''' Main
'''
if __name__ == "__main__":
	knn = KNN(['P1', 'P2'])	
	knn.addDataSet({'P1':[1.0, 1.1, 0.1, 0.2], 
						'P2':[1.1, 1.0, 2.0, 2.1],
						'Class': ['A', 'A', 'B', 'B']})
	knn.addDataSet({'P1':[1.0, 3.1, 0.2, 0.3], 
							'P2':[1.2, 0.0, 7.0, 4.1],
							'Class': ['A', 'A', 'B', 'B']})					
	
	print(knn.classify({'P1':1.1, 'P2':1.0}, 5))