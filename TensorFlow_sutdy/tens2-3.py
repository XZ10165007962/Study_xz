import tensorflow as tf
import numpy as np

#fx = a*x**2 + b*x +c 的导数

x = tf.Variable(0.0,name='x',dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

with tf.GradientTape() as tape:
    y = a*tf.pow(x,2) + b * x + c

dy_dx = tape.gradient(y,x)
print(dy_dx)

with tf.GradientTape() as tape:
    tape.watch([a,b,c])
    y = a * tf.pow(x, 2) + b * x + c

dy_dx,dy_da,dy_db,dy_dc = tape.gradient(y,[x,a,b,c])
print(dy_dx)
print(dy_da)
print(dy_db)
print(dy_dc)

with tf.GradientTape() as tape1:
    with tf.GradientTape() as tape2:
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape2.gradient(y,x)
dy2_dx2 = tape1.gradient(dy_dx,x)
print(dy2_dx2)

@tf.function
def f(x):
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)

    x = tf.cast(x,tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape.gradient(y,x)
    return (dy_dx,y)

tf.print(f(tf.constant(0.0)))
tf.print(f(tf.constant(1.0)))

x = tf.Variable(0.0,name='x',dtype=tf.float32)

def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a*tf.pow(x,2)+b*x+c
    return y
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(1000):
    optimizer.minimize(f,[x])

tf.print('y=',f(),'x=',x)


a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)
x = tf.Variable(0.0,name='x',dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(1000):
    with tf.GradientTape() as tape:
        y = a*tf.pow(x,2)+b*x+c
    dy_dx = tape.gradient(y,x)
    optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])

tf.print('y=',f(),'x=',x)

x = tf.Variable(0.0,name='x',dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
@tf.function
def minimzef():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)

    for _ in tf.range(1000):
        with tf.GradientTape() as tape:
            y = a * tf.pow(x, 2) + b * x + c
        dy_dx = tape.gradient(y,x)
        optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])

    y = a * tf.pow(x, 2) + b * x + c
    return y
tf.print(minimzef())
tf.print(x)

x = tf.Variable(0.0,name = "x",dtype = tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

@tf.function
def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a*tf.pow(x,2)+b*x+c
    return(y)

@tf.function
def train(epoch):
    for _ in tf.range(epoch):
        optimizer.minimize(f,[x])
    return(f())


tf.print(train(1000))
tf.print(x)
