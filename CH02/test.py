#!/usr/bin/python3
#-*- coding:utf-8 -*-

import matplotlib.pyplot as plt
from KNN import *

FN_DATING_DATA = 'datingTestSet.txt'

'''绘制散点图'''
def draw(x, y, color='b', size=20):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	#设置轴标签  
	plt.xlabel(x.name)  
	plt.ylabel(y.name)
	ax.scatter(x, y, c=color, s=size, marker='o')
	plt.show()
	return


''' Test1，简单测试KNN算法 '''
def KnnTest():
	knn = KNN(['P1', 'P2'])	
	knn.addDataSet({'P1':[1.0, 1.0, 0.0, 0.0], 
					'P2':[1.1, 1.0, 0.0, 0.1],
					'Class': ['A', 'A', 'B', 'B']})
	knn.addDataSet({'P1':[1.0, 3.1, 0.2, 0.3], 
						'P2':[1.2, 0.0, 0.0, 0.1],
						'Class': ['A', 'A', 'B', 'B']})					
	print(knn.classify({'P1':0.0, 'P2':0.2}, 3))
	del(knn)
	return 


''' Test2，从文件读取数据并用散点图显示'''
def ShowData():
	dating = KNN()
	dating.getDataSetByFile(fn= FN_DATING_DATA,
			sep='[\s,\t,\,]+', 
			names=['Flying','TVgame','IceCream','Class'])
	#根据分类生成数字指代的分类表，并绘制图像
	classList = dating.dataSet[CLASS_COL_NAME].replace({"largeDoses":1, "smallDoses":2, "didntLike":3})
	draw(dating.dataSet['IceCream'], dating.dataSet['Flying'], color=30*classList)
	del(dating)
	return


''' Test3, 从文件读取数据，归一化并测试分类器错误率 '''
def DatingClassTest():
	dating = KNN()
	# 读取数据
	dating.getDataSetByFile(fn= FN_DATING_DATA,
				sep='[\s,\t,\,]+', 
				names=['Flying','TVgame','IceCream','Class'])
	# 归一化数据
	dating.autoNorm()
	# 取出10%的数据作为测试验证
	(testDataSet, target) = dating.getSamples(fraction=0.1)	
	count = 0
	fail = 0
	# 验证每一个测试数据
	for i in testDataSet.index:
		count = count +1
		#print("i=%d, target=%s\n" %(i, target[i]))
		#print(dict(testDataSet.loc[i]))
		res = dating.classify(dict(testDataSet.loc[i]), k=3)
		if res == target[i]:
			#print("success, res=%s" %res)
			pass
		else:
			fail = fail +1
			print("fail, index=%d, res=%s, should be %s" %(i, res,target[i]))
			print(testDataSet.loc[i])
	print("Fail：%d/%d, %.2f%%\n" %(fail, count, float(fail)/float(count)*100 ) )
	del(dating)
	return



''' Test4, 从文件读取数据，归一化并测试输入数据 '''
def ClassifyPerson(Flying, TVgame, IceCream ):
	dating = KNN()
	# 读取数据
	dating.getDataSetByFile(fn= FN_DATING_DATA,
				sep='[\s,\t,\,]+', 
				names=['Flying','TVgame','IceCream','Class'])
	#输入测试数据
	test = {'Flying':[Flying],'TVgame':[TVgame],'IceCream':[IceCream]}
	dating.addDataSet(test)
	# 归一化数据
	dating.autoNorm()
	# 分类
	#print(dating.getDataSet().head(dating.count-1) )
	res = dating.classify(dict(dating.getDataSet().loc[dating.count-1]), k=3, data=dating.getDataSet().head(dating.count-1))
	print(res)
	del(dating)
	return



''' Main
'''
if __name__ == "__main__":
	#KnnTest()
	#ShowData()
	DatingClassTest()
	#ClassifyPerson(Flying=10000, TVgame=10, IceCream=0.5)
