#!/usr/bin/python3
#-*- coding:utf-8 -*-

from KNN import *
import os

DIR_TRAINING_DATA = "digits/trainingDigits/"
DIR_TEST_DATA = "digits/testDigits/"
#DIR_TEST_DATA = "digits/mytest/"
EACH_DATA_SIZE = 1024

'''将txt数据读取为数组'''
def img2verctor(dir, fn):
	# 读取文件所有内容
	with open(dir+fn, 'r') as f:
			data = f.readlines()
	res=[]
	# 遍历文件内容，转换为数组
	for i in data:
		for j in i:
			if j!='\n':
				res += [int(j)]
	# 根据文件名获得目标分类
	target = fn.split('_')[0]
	return res, target


'''从文件中获取数据，生成dataframe数据结构
training == True，获取训练数据，包括分类结果
training == False，获取测试数据，理想分类结果放在target中
'''
def getData(dir, training = True):
	numClass = []
	temp = []
	# 遍历文件夹内的所有文件
	for i in os.listdir(dir):
		if i.split('.')[1]!='txt':
			continue
		# 将文件转换为特征数据组
		(data, target) = img2verctor(dir,i)
		numClass += [target]
		temp += [data]
	# 根据N组特征数据生成新的dataframe，以便于提供给KNN类计算
	data=pd.DataFrame(temp)
	if training:
		data[CLASS_COL_NAME]=numClass
		return data
	else:
		return data,numClass
		

''' Main
'''
if __name__ == "__main__":
	num = KNN()
	count = 0
	fail = 0
	# 读取训练数据，并set给KNN类
	num.setDataSet(getData(DIR_TRAINING_DATA, training=True))
	
	# 验证每一个测试数据
	'''方式一：全部获得测试数据后计算'''
	(testDataSet, target) = getData(DIR_TEST_DATA, training=False)
	for i in testDataSet.index:
		count = count +1
		#print("i=%d, target=%s\n" %(i, target[i]))
		#print(testDataSet.loc[i])
		res = num.classify(dict(testDataSet.loc[i]), k=3)
		if res == target[i]:
			#print("success, res=%s" %res)
			pass
		else:
			fail = fail +1
			print("fail, count=%d, res=%s, should be %s" %(count, res,target[i]))
			#print(testDataSet.loc[i])
	
	'''方式二，遍历每个文件单独计算，可获得当前文件名(调试用)
	lists = range(EACH_DATA_SIZE)
	for i in os.listdir(DIR_TEST_DATA):
		if i.split('.')[1]!='txt':
			continue
		(data, target) = img2verctor(DIR_TEST_DATA,i)
		count = count +1
		print("i=%s, target=%s" %(i, target))
		print(data)
		res = num.classify(dict(zip(lists, data)), k=3)
		if res == target:
			#print("%i success, res=%s" %(i, res))
			pass
		else:
			fail = fail +1
			print("fn=%s fail, count=%d, res=%s, should be %s" %(i, count, res,target))
			#print(data)
	'''
	print("Fail：%d/%d, %.2f%%\n" %(fail, count, float(fail)/float(count)*100 ) )
	del(num)
