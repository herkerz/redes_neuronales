import pandas as pd
import numpy as np
from red_neuronal import red_neuronal, red_neuronal_sin_bias


## Parametros
maxit = 4000
lr = 0.4

rnn = red_neuronal(2,10,1,lr)

x1     = [0,0,1,1]
x2     = [0,1,0,1]
target = [0,1,1,0]
data = {'x1' : x1, 'x2' : x2, 'target' : target}
data = pd.DataFrame(data)

x = data.drop('target',axis=1).to_numpy()
y = data['target'].to_numpy()


rnn.train(x,y,maxit = maxit)


print("Con bias")
print(rnn.predict(np.array([0,0])))
print(rnn.predict(np.array([1,1])))
print(rnn.predict(np.array([0,1])))
print(rnn.predict(np.array([1,0])))

print("########################3")

rnn_sin_bias = red_neuronal_sin_bias(2,10,1,lr)

rnn_sin_bias.train(x,y,maxit = maxit)
print("\nSin bias")
print(rnn_sin_bias.predict(np.array([0,0])))
print(rnn_sin_bias.predict(np.array([1,1])))
print(rnn_sin_bias.predict(np.array([0,1])))
print(rnn_sin_bias.predict(np.array([1,0])))