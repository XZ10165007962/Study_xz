import tensorflow as tf
import datetime
import os
'''x = tf.constant('hello')
y = tf.constant('world')
z = tf.strings.join([x,y],separator=' ')
tf.print(z)'''

start = datetime.datetime.now()
def strjoin(x,y):
    z = tf.strings.join([x,y],separator=' ')
    tf.print(z)
    return z

result = strjoin(tf.constant("hello"),tf.constant("world"))
end = datetime.datetime.now()
print(result)
print(end - start)

'''start = datetime.datetime.now()
@tf.function
def strjoin(x,y):
    z = tf.strings.join([x,y],separator=' ')
    tf.print(z)
    return z

result = strjoin(tf.constant("hello"),tf.constant("world"))
end = datetime.datetime.now()
print(result)
print(end - start)'''
