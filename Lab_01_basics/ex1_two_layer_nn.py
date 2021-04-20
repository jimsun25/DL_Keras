import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def F(x1, x2):
  return np.sin(np.pi * x1 / 2.0) * np.cos(np.pi * x2 / 4.0)
  
A = 2
nb_samples = 1000
X_train = np.random.uniform(-A, +A, (nb_samples, 2))
Y_train = np.vectorize(F)(X_train[:,0], X_train[:,1])

model = Sequential()

nb_neurons = 20
model.add(Dense(nb_neurons, input_shape=(2,)))
model.add(Activation('relu'))
model.add(Dense(1))

# Shortcut
#model = Sequential([Dense(nb_neurons, input_shape=(2,)),
#                    Activation('relu'), Dense(1)])

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_train, Y_train, epochs=10, batch_size=32)

x = [1.5, 0.5]
print(F(x[0], x[1]))
x = np.array(x).reshape(1, 2)
print(x)
print( model.predict(x) )
print( model.predict(x)[0][0] )

Width = 200
Height = 200
U = np.linspace(-A, +A, Width)
V = np.linspace(-A, +A, Height)

# Computes cartesian product between U and V:
UV = np.transpose([np.tile(U, len(V)), np.repeat(V, len(U))])
print(UV)
ys = model.predict(UV)
print(ys)
I = ys.reshape(Width, Height)

#Make imageplotlib show the images
plt.imshow(I, cmap = cm.Greys)
plt.show()

