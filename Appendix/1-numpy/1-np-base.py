#!/usr/bin/env python
# pylint: disable=C0103

"""
numpy api simple demo
"""
import numpy as np

# numpy.ndarray 的主要属性
###################################################
print("#" * 30)
print("ndarray的主要属性")
arr = np.array([1, 2, 3])
print("元素类型:", arr.dtype)
print("数组的秩(维数):", arr.ndim)  # 数组的秩(维数): 1
print("每一维的长度:", arr.shape)
print("元素总数:", arr.size)
print("每个元素的byte数:", arr.itemsize)
print("数组的实际内存地址:", arr.data)

arr2 = np.array([[1, 2, 3], [1, 2, 3]])
print("元素类型:", arr2.dtype)
print("数组的秩(维数):", arr2.ndim)  # 数组的秩(维数): 2
print("元素总数:", arr2.size)
print("每个元素的byte数:", arr2.itemsize)
print("数组的实际内存地址:", arr2.data)

arr3 = np.array([[1, 2, 3], [1, 2]])
print("元素类型:", arr3.dtype)
print("数组的秩(维数):", arr3.ndim)  # 数组的秩(维数): 1 元素的数据类型为list
print("元素总数:", arr3.size)
print("每个元素的byte数:", arr3.itemsize)  # 类型为object 为什么元素的size也是8？
print("数组的实际内存地址:", arr3.data)

arr4 = np.array([
    [[1, 2, 3], [1, 2, 3]],
    [[1, 2, 3], [1, 2, 3]]
])
print("元素类型:", arr4.dtype)
print("数组的秩(维数):", arr4.ndim)  # 数组的秩(维数)：3 元素的数据类型为list
print("每一维的长度:", arr4.shape)
print("元素总数:", arr4.size)
print("每个元素的byte数:", arr4.itemsize)
print("数组的实际内存地址:", arr4.data)


# numpy.ndarray 的创建方式
###################################################
print("#" * 30)
print("使用numpy.array函数创建数组")
arr = np.array([1, 2, 3, 4, 5])
# 错误的创建方法: arr1 = np.array(1,2,3,4,5)
print("数组:", arr)

arr = np.array([(1, 4, 5), (4, 5, 6)])  # 使用序列创建ndarray
print("数组:", arr)

arr = np.array([1, 2], dtype=complex)  # 创建时显示指定元素类型
# 下面是错误的创建方式，第一个元素是list，不符合指定的类型
# arr1 = np.array([[1, 2], 2], dtype=complex)
print("数组:", arr)

# 使用zeros,ones和empty函数创建数组指定维度和各维长度的数组，元素类型都是float64
print("#" * 30)
print("使用zeros,ones,empty函数创建数组")
arr = np.zeros((3, 5))  # 创建3行5列的数组，数组所有元素都是0
print("数组:", arr)

arr = np.ones((3, 5))  # 创建3行5列数组，数组所有元素都是1
print("数组:", arr)

arr = np.empty((3, 5))  # 创建数组，数组元素不确定
print("数组:", arr)

# 规定元素取值范围的开始值，结束值(不包含)以及步长来创建数组
print("#" * 30)
print("使用arange,linspace函数创建数组")
arr = np.arange(10, 30, 5)  # start <end,如果此时步长为-5将创建一个size=0的书组
print("数组:", arr)  # 创建数组，
arr = np.arange(30, 10, -5)  # start > end,如果此时步长为5,将创建一个size=0的数组
print("数组:", arr)
# arange的步长参数可以接受浮点数，但是此时创建的元素个数无法确定
arr = np.arange(0, 2, 0.3)
print("数组:", arr)

# 规定元素取值范围的开始值,结束值(不包含)以及元素个数来创建数组
arr = np.linspace(0, 3, 9)
print("数组:", arr)

# 使用随机数创建数组
arr = np.random.random((5, 4))
print("随机数组", arr)


# 数组打印格式
###################################################
print("#" * 30)
print("数组的打印格式")
# 一维数组，从左到右打印到一行中
print("数组:", np.arange(6))
# 二维数组，第一维从上到下打印，第二维从左到右打印到一行中
print("数组:", np.arange(12).reshape(4, 3))
# 最后一维从左到右打印到一行，倒数第二维从上倒下，其他维度从上到下打印，第一维中的每个多维数组之间使用空行分隔
# 初第一维外,每一维比上一维在打印时向内缩进一个空格
print("数组:", np.arange(24).reshape(2, 3, 4))
# 打印过长数组时，中间元素用...代替
print(np.arange(10000))
# 多为数组，中间维度用...代替
print(np.arange(10000).reshape(100, 100))

# 数组的数学计算
###################################################
print("#" * 30)
print("数组的数学计算")
a = np.array([20, 30, 40, 50])
b = np.arange(4)
print("数组a:", a)
print("数组b:", b)
# 数组加法
print("数组a+b:", a + b)
# 数组的n次幂
print("数组b**2:", b**2)
# 对数组求sin,数组余常数的乘法
print("数组10*np.sin(a):", 10 * np.sin(a))
# 数组与常数比较,数组与数组比较
print("数组a<35:", a < 35)
print("数组a<b", a < b)

A = np.array([[1, 1], [0, 1]])
B = np.array([[2, 0], [3, 4]])
# 数组的乘法
print("数组A*B:", A * B)  # 数组对应位置的元素相乘
# 对数组做矩阵乘法
print("数组作为矩阵相乘A·B(1)：", A.dot(B))
print("数组作为矩阵相乘A·B(2):", np.dot(A, B))
# 数组的+=，*=运算
a = np.ones((2, 3), dtype=int)
b = np.random.random((2, 3))
print("数组a:", a)
print("数组b:", b)
a *= 3
print("数组a(执行a*=3后):", a)
b += a
print("数组b(执行b+=a后):", b)
# a += b # 错误的运算，b不能从float自动转换成int。
# 执行a+=b是解释器会提示一下错误信息
# Cannot cast ufunc add output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
# 但在python2.7.9中不会出错


# 数组运算中的类型自动转换
# 不同类型的数组进行运算，得到的结果数组的类型与更通用或者更精确的类型相同
print()
print("数组", a + b)  # 结果数组中元素的类型是float
a = np.ones(3, dtype=np.int32)
b = np.linspace(0, np.pi, 3)
print("数组a, 类型为:", a.dtype.name, "\r\n", a)
print("数组b, 类型为:", b.dtype.name, "\r\n", b)
c = a + b
print("数组c, 类型为:", c.dtype.name, "\r\n", c)
d = np.exp(c * 1j)
print("数组d, 类型为:", d.dtype.name, "\r\n", d)

# 一些一元运算作为ndarray类的方法实现，例如累加和(sum),最小值(min)，最大值(max)
print()
a = np.random.random((2, 3))
print("数组a:", a)
print("数组a的累加和:", a.sum())
print("数组a在各个元素上的累加和:", a.cumsum())
print("数组a中元素的最小值:", a.min())
print("数组a中的最大值:", a.max())
# 默认情况下，这些一元运算方法与ndarray的维度无关，只将ndarray作为所有元素的一个列表处理。
# 当指定axis参数时,方法将按照axis进行计算
a = np.arange(12).reshape(3, 4)
print("数组a:", a)
print("数组a在第一维上求累加和:", a.sum(axis=0))   # 按列求累加和
print("数组a在第二维上求累加和:", a.sum(axis=1))   # 按行求累加和
print("数组a在第二维上求最大值:", a.max(axis=1))   # 按行求最大值
print("数组a在第一维上各个元素上的累加和:", a.cumsum(axis=0))  # 按列求各元素的累加和
print("数组a在第二维上各个元素上的累加和:", a.cumsum(axis=1))  # 按行求各元素的累加和

# numpy 中的通用数学函数(Function),如sin,cos,exp,sqrt等,实例化的ndarray对象没有与之对应的方法(Method)
# 注:在python中,Function和Method是不同的,参考http://www.cnblogs.com/blackmatrix/p/6847313.html
print("#" * 30)
print("numpy中的通用数学函数")
A = np.arange(3)
print("数组A:", A)
print("数组np.exp(A):", np.exp(A))
# print("数组A.exp():", A.exp()), ndarray没有exp方法
print("数组np.sqrt(A):", np.sqrt(A))
# print("数组A.sqrt():", A.sqrt()),ndarray没有sqrt方法
print("数组np.sin(A):", np.sin(A))
# print("数组A.sin():", A.sin()),ndarray没有sin方法

# ndarray的索引,切片和迭代
print("#" * 30)
print("ndarray的索引,切片和迭代")
# 对于一维数组,他的索引,切片和迭代与list相同
a = np.arange(10)**3
print("数组a:", a)
print("元素a[2]:", a[2])
print("数组切片a[2:5]", a[2:5])
a[:6:2] = -1000  # 从数组的第0各元素开始,每隔2-1个元素赋值为-1000,到第6-1各元素结束
print("数组a:", a)
print("数组a的倒序:", a[::-1])
print("数组a的迭代:")
for i in a:
    print(i**(1 / 3.))


# 对于多为数组,每个维度都有对应的索引,整体的索引由一个用逗号分隔的序列给出
# 每一维的索引符合list的索引,切片规则
# 注意:a(x,y)这种写法是错误的,python解释器会认为是函数调用,可以使用a[(x,y)]

def func(x, y):
    """
    根据索引创建ndarray数组
        :param x: row index
        :param y: column index
    """
    return 10 * x + y


a = np.fromfunction(func, (5, 4), dtype=int)
print("数组a:", a)
print("数组a中第2行,第3列的元素a[2,3]:\r\n", a[2, 3])
print("数组a中的第4行第3列元素a[(3,4)]:\r\n", a[(4, 3)])
print("数组中每一行第1列的元素a[0:5,1]:\r\n", a[0:5, 1])
print("数组中每一行第1列的元素a[:, 1]:\r\n", a[:, 1])
print("数组中每一行第1列的元素a[:5,1]:\r\n", a[:5, 1])
print("数组中每一行第1列的元素a[0:,1]:\r\n", a[0:, 1])
print("数组中第一行和第二行的每一列a[1:3,:]:\r\n", a[1:3, :])
print("数组中第一行和第二行的每一列a[1:3,:]:\r\n", a[1:3, 0:])
print("数组中第一行和第二行的每一列a[1:3,:]:\r\n", a[1:3, 0:4])
print("数组中第一行和第二行的每一列a[1:3,:]:\r\n", a[1:3, :4])
print("数组中第一行和第二行的每一列a[1:3,:]:\r\n", a[1:3])
print("数组中第一行和第二行中的第一列到第三列a[1:3,1:4]:\r\n", a[1:3, 1:4])

# 多维数组的索引中"..."的含义
# 索引中只能有一个"..."
a = np.arange(4 * 5 * 4 * 3).reshape((4, 5, 4, 3))
print("数组a:\r\n", a)
print("三维数组a[1,...]以及他的shape:", a[1, ...].shape, "\r\n", a[1, ...])
print("三维数组a[...,1]以及他的shape:", a[..., 1].shape, "\r\n", a[..., 1])
# print("数组中第3维a[...,4,...]", a[...,4,...]) 这是错误的
print("二维数组a[1,...,2]以及他的shape:", a[1, ..., 2].shape, "\r\n", a[1, ..., 2])
print("一维数组a[1,...,1,2]:\r\n", a[1, ..., 1, 2])
print("三维数组a[:,2,...]以及他的shape:", a[:, 2, ...])

# 多维数组的迭代
print("默认情况下遍历多维数组的第0个维度")
index = 0
for i in a:
    print("第0个维度中的第%d个三维数组以及它的shape:" % (index), i.shape, "\r\n", i)
    index += 1

print("使用ndarray的flat属性遍历多维数组中的每一个元素")
print(np.array([element for element in a.flat]))


# 对于shape的控制,控制数组的维数以及各维的长度
print("#" * 30)
print("对于shape的控制,也就是控制数组的维数以及各维的长度")
# shape 由一个整数序列指定,序列中的元素个数表示数组的维数,每个元素的数值表示对应维度的长度
# 序列中表示各元素位置的下标序号代表维度号.如(2,3)代表一个2维数组,第0维长度为2,第1维长度为3
# 创建ndarray时可以指定shape,
a = np.floor(10 * np.random.random([3, 4]))
print("数组a和它的的shape:", a.shape, "\r\n", a)
# 可以调用ndarray对象的reshape,ravel两个方法以及T,获得一个在ndarry基础上的不同shape的新ndarray
b = a.reshape(2, 2, 3)  # 返回一个与a的元素相同,shape=(2,,2,3)的新ndarray
print("数组b和它的的shape:", b.shape, "\r\n", b)
print("a扁平化后的数组", a.ravel())  # a.revel() 返回一个a扁平化后的一维数组
print("a转置后的数组和它的shape", a.T.shape, "\r\n", a.T)
print("原数组a和它的的shape没有变化:", a.shape, "\r\n", a)
# b = a.reshape(2,4) 错误的shape转换,目标数组的size与原数组的size不同.

# 修改数组自身的shape
a = np.arange(12).reshape(2, 6)
print("数组a以及它的shape:", a.shape, "\r\n", a)
b = np.resize(a, (3, 4))
print("新数组b以及他的shape:", b.shape, "\r\n", b)
print("原数组a和它的shape没有发生变化", a.shape, "\r\n", a)
b = a.resize((3, 4))
print("方法a.resize((3,4))没有返回值:", b)
print("原数组a和它的shape发生变化", a.shape, "\r\n", a)
a.shape = (2, 2, 3)
print("也可以通过a.shape=(2,2,3),修改a的shape:", a.shape, "\r\n", a)

# 堆叠不同的数组
print("#" * 30)
print("堆叠不同的数组")
print("使用函数vstack和hstack堆叠")
a = np.floor(10 * np.random.random((2, 3)))
b = np.floor(10 * np.random.random((2, 3)))
c = np.floor(10 * np.random.random((2, 2)))
e = np.floor(10 * np.random.random((2, 2, 3)))
f = np.floor(10 * np.random.random((2, 2, 3)))
m = np.vstack((a, b))
print("a和b在第0维上堆叠(其他维度的长度不变):", m.shape, "\r\n", m)
m = np.hstack((a, b))
print("a和b在第1维上堆叠(其他维度的长度不变):", m.shape, "\r\n", m)
# m = np.vstack((a, c)) 这种堆叠是错误的,除第一维以外的各对应维度的长度不能完全相等
m = np.hstack((a, c))
print("a和c进行横向堆叠(不改变每一列的长度):", m.shape, "\r\n", m)
# m = np.hstack((c, e)) 这种堆叠是错误的,输入的两个数组的维数不同,
m = np.hstack((e, f))
print("e和f在第0维上堆叠(其他维度的长度不变)", m.shape, "\r\n", m)
m = np.vstack((e, f))
print("e和f在第1维上堆叠(其他维度的长度不变)", m.shape, "\r\n", m)

# 使用column_stack堆叠,column_stack相当于hstack
print()
print("使用column_stack堆叠")
a = np.arange(5)
b = np.arange(5)
print("数组a:", a.shape, "\r\n", a)
print("数组b:", b.shape, "\r\n", b)
print("数组a.T:", a.T.shape, "\r\n", a.T)
print("数组b.T:", b.T.shape, "\r\n", b.T)
m = np.column_stack((a, b))
print("a和b堆叠:", m.shape, "\r\n", m)
m = np.column_stack((a.T, b.T))
print("a.T和b堆.T叠:", m.shape, "\r\n", m)

a = np.arange(10).reshape(5, 2)
b = np.arange(15).reshape((5, 3))
print("数组a:", a.shape, "\r\n", a)
print("数组b:", b.shape, "\r\n", b)
print("数组a.T:", a.T.shape, "\r\n", a.T)
print("数组b.T:", b.T.shape, "\r\n", b.T)
m = np.column_stack((a, b))
print("a和b堆叠:", m.shape, "\r\n", m)
# m = np.column_stack((a.T, b.T)) 这是错误的,转置后,shape分别为(2,5),(3,5),列的(column)的长度不同
c = np.arange(12).reshape(2, 2, 3)
d = np.arange(12).reshape(2, 2, 3)
m = np.column_stack((c, d))
print("c和d堆叠:", m.shape, "\r\n", m)
m = np.column_stack((c.T, d.T))
print("c.T和d.T堆叠:", m.shape, "\r\n", m)
e = np.arange(24).reshape(2, 2, 3, 2)
f = np.arange(24).reshape(2, 2, 3, 2)
m = np.column_stack((e, f))
print("e和f堆叠:", m.shape, "\r\n", m)
m = np.column_stack((e.T, f.T))
print("e.T和f.T堆叠:", m.shape, "\r\n", m)
