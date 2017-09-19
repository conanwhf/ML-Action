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
CLASS_COL_NAME = 'Class'
NORM_COL_PLUS = '_Norm'

class KNN(object):
	def __init__(self, labels=[]):#特征点的属性名称
		self.dataSet = pd.DataFrame(columns=labels+ [CLASS_COL_NAME])
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
		self.dataSet = data.copy(deep=True)
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
		res = data.head(k)[CLASS_COL_NAME].value_counts()
		#print(res)
		return res.index[0]
	
	# 数据归一化，将所有特征值根据取值范围归一化为0-1的数据，以消除不同特征数据大小对欧式距离计算造成的影响
	def autoNorm(self):
		# 1. 获取每个特征点的最大最小值
		minVals = self.dataSet.drop(CLASS_COL_NAME, 1).min(0)
		maxVals = self.dataSet.drop(CLASS_COL_NAME, 1).max(0)
		ranges = maxVals - minVals
		#print(ranges)
		new = pd.DataFrame()
		# 2. 分别对每个特征点数据进行归一化，跳过分类结果列
		for i in self.dataSet:
			if i!= CLASS_COL_NAME:
				new[i+NORM_COL_PLUS]=self.dataSet[i]/ranges[i]
				self.dataSet.drop([i], inplace=True,axis=1)
		# 3. 将归一化后的数据直接添加至原有数据表
		#self.dataSet = pd.merge(self.dataSet, new, how='outer', left_index=True,right_index=True)
		self.dataSet = pd.concat([self.dataSet,new],axis=1)
		#print(self.dataSet)
		return new
	
	# 将部分数据取出作为验证组，并从原数据集中删除
	def getSamples(self, fraction=0.1):
		# 1. 抽样(随机或头n个)
		samples = self.dataSet.sample(n=None, frac=fraction, replace=False, weights=None, random_state=None, axis=0)
		#samples = self.dataSet.head(int(self.count*fraction))
		# 2. 将抽样数据从训练集中删除
		self.dataSet.drop(samples.index, inplace=True, axis=0)
		# 3. 将目标结果独立抽出
		target = samples[CLASS_COL_NAME]
		samples = samples.drop(CLASS_COL_NAME, axis=1)
		#print(self.dataSet)
		return samples, target
		
		
		