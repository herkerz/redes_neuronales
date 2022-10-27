import numpy as np
import pandas as pd

class red_neuronal:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate
        
    def initial_weights(self):
        self.weights_ih = np.random.rand(self.hidden_nodes, self.input_nodes)
        self.weights_ho = np.random.rand(self.output_nodes, self.hidden_nodes)
        
        self.bias_h = np.random.rand(self.hidden_nodes, 1)
        self.bias_o = np.random.rand(self.output_nodes, 1)
        
    @staticmethod    
    def relu(weigthed_activation):
        return np.maximum(0, weigthed_activation)
    
    @staticmethod    
    def sigmoid(weigthed_activation):
        return 1 / ( 1 + np.exp(-weigthed_activation))
    
    def forward_prop(self):
        self.hidden_input = self.weights_ih.dot(self.input_vector) + self.bias_h
        self.hidden_output = red_neuronal.sigmoid(self.hidden_input)
        
        self.output = self.weights_ho.dot(self.hidden_output) + self.bias_o

    
    def back_prop(self):

        self.error_output = np.subtract(self.targets_array,self.output.T).T

        self.error_hidden = self.weights_ho.T.dot(self.error_output)
        
        gradients = self.lr * self.error_output 
        self.deltas_hidden_output = gradients.dot(self.hidden_output.T)
        
        self.weights_ho = np.add(self.weights_ho, self.deltas_hidden_output)
        self.bias_o = self.bias_o + gradients
        
        gradients_hidden = self.lr * self.error_hidden * (self.hidden_output * (1 - self.hidden_output) )
        self.deltas_input_hidden = gradients_hidden.dot( self.input_vector.T)
        
        self.weights_ih = np.add(self.weights_ih, self.deltas_input_hidden)
        self.bias_h = self.bias_h + gradients_hidden
        
    def iterate(self,input_array, targets_array,  num_iterations):
        
        for i in range(num_iterations):
            rnd_idx = np.random.choice(range(len(input_array)))
            rnd_row = input_array[rnd_idx,:]
            rnd_tgt = targets_array[rnd_idx]
            
            self.input_vector = rnd_row.reshape((len(rnd_row),1))
            self.targets_array = rnd_tgt
        
            self.forward_prop()
            self.back_prop()

        
    def train(self, input_array, targets_array, maxit = 10000 ):
    
        self.initial_weights()
        self.iterate(input_array, targets_array, maxit)
        
    def predict(self, inputs):
        
        inputs = inputs.reshape((len(inputs),1))
        hidden_input = self.weights_ih.dot(inputs) + self.bias_h
        hidden_output = red_neuronal.sigmoid(hidden_input)
        
        prediction = self.weights_ho.dot(hidden_output) + self.bias_o

        
        return prediction
    
rnn = red_neuronal(2,10,1,0.2)

x1     = [0,0,1,1]
x2     = [0,1,0,1]
target = [0,1000,1000,0]
data = {'x1' : x1, 'x2' : x2, 'target' : target}
data = pd.DataFrame(data)

x = data.drop('target',axis=1).to_numpy()
y = data['target'].to_numpy()

maxit = 20000
rnn.train(x,y,maxit = maxit)

print(rnn.predict(np.array([0,0])))
print(rnn.predict(np.array([1,1])))
print(rnn.predict(np.array([0,1])))
print(rnn.predict(np.array([1,0])))