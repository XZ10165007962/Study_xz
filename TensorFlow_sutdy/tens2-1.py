import numpy as np
import tensorflow as tf

i = tf.constant(1)
l = tf.constant(1,dtype=tf.int64)
f = tf.constant(1.23)
d = tf.constant(1.23,dtype=tf.double)
s = tf.constant('hello world')
b = tf.constant(True)

print(tf.int64 == np.int64)
print(tf.bool == np.bool)
print(tf.double == np.float64)
print(tf.string == np.unicode) # tf.string类型和np.unicode类型不等价

scalar = tf.constant(True)  #标量，0维张量

print(tf.rank(scalar))
print(scalar.numpy().ndim)

vector = tf.constant([1.0,2.0,3.0,4.0])

print(tf.rank(vector))
print(vector.numpy())

matrix = tf.constant([[1.0,2.0],[3.0,4.0]])
print(tf.rank(matrix).numpy())
print(np.ndim(matrix))

tensor3 = tf.constant([[[1.0,2.0],[3.0,4.0]],[[5.0,6.0],[7.0,8.0]]])  # 3维张量
print(tensor3)
print(tf.rank(tensor3))

h = tf.constant([123,456],dtype=tf.int32)
f = tf.cast(h,tf.float32)
print(h.dtype,f.dtype)

u = tf.constant(u"你好 世界")
print(u.numpy())
print(u.numpy().decode("utf-8"))


c = tf.constant([1.0,2.0],name='c')
print(c)
print(id(c))
c = c + tf.constant([1.0,1.0])
print(c)
print(id(c))


# 变量的值可以改变，可以通过assign, assign_add等方法给变量重新赋值
v = tf.Variable([1.0,2.0],name = "v")
print(v)
print(id(v))
#v.assign_add([1.0,1.0])
v.assign([3.0,4.0])
print(v)
print(id(v))