import numpy as np
import math

class Layer:

    act_funcs= ['relu', 'sigmoid']
    init_methods = ['standard', 'xavier', 'he']

    def __init__(self, n_curr, n_prev, act_func, init_method = 'standard'):
        assert act_func in self.act_funcs
        assert init_method in self.init_methods

        self.b = np.zeros((n_curr,1))
        self.W = np.random.randn(n_curr,n_prev)
        self.Z = None
        self.A = None
        self.dA = None
        self.dW = None
        self.db = None

        #for use in dropout
        self.D = None       

        #for momemtum
        self.vdW = np.zeros(self.W.shape)     
        self.vdb = np.zeros(self.b.shape)
        
        #for RMS prop
        self.sdW = np.zeros(self.W.shape)
        self.sdb = np.zeros(self.b.shape)

        self.init_method = init_method
        if init_method == 'standard':
            self.W = self.W * 0.01
        elif init_method == 'xavier':
            self.W = self.W / np.sqrt(n_prev)
        elif init_method == 'he':
            self.W = self.W * np.sqrt(2/n_prev)

        assert self.W.shape == (n_curr, n_prev)
        assert self.b.shape == (n_curr, 1)

        self.act_func = act_func
        if act_func == 'sigmoid':
            self._act_fwd = Layer._sig_fwd
            self._act_bwd = Layer._sig_bwd
        elif act_func == 'relu':
            self._act_fwd = Layer._relu_fwd
            self._act_bwd = Layer._relu_bwd


    def forward_prop(self, A_prev, keep_prob):
        assert(self.W.shape[1] == A_prev.shape[0])

        self.Z = self._linear_fwd(A_prev)
        self.A = self._act_fwd(self.Z)

        if keep_prob < 1:
            self.D = np.random.rand(self.A.shape[0],self.A.shape[1])
            self.D = np.int8(self.D < keep_prob)
            self.A *= self.D
            self.A /= keep_prob

        assert(self.A.shape[1] == A_prev.shape[1])
        assert(self.A.shape[0] == self.W.shape[0])

        return self.A

    def backward_prop(self, dA, A_prev, keep_prob, lambd):
        assert (self.A.shape == dA.shape)

        if keep_prob < 1:
            dA *= self.D
            dA /= keep_prob

        dZ = np.multiply(dA,self._act_bwd(self.Z))
        dA_prev = self._linear_bwd(dZ, A_prev, lambd)

        assert(dA_prev.shape == (self.dW.shape[1],dA.shape[1]))

        return dA_prev

    def update_params(self, learning_rate, optimizer, beta1, beta2, epsilon, t):
        
        if optimizer == 'gd':
            self.W = self.W - (learning_rate * self.dW)
            self.b = self.b - (learning_rate * self.db)
        
        else: #if optimizer is 'adam' or 'momentum'
            #make dW an exponential weighted average of dW (implement momemtum)
            self.vdW = beta1 * self.vdW + (1-beta1) * self.dW
            vdW_corr = self.vdW / (1-beta1**t)
            
            self.vdb = beta1 * self.vdb + (1-beta1) * self.db
            vdb_corr = self.vdb / (1-beta1**t)

            #Penalize dW by dividing sqrt of exponential weighted average of dW**2 (implement RMSprop)
            if optimizer == 'adam':
                self.sdW = beta2 * self.sdW + (1-beta2) * self.dW ** 2
                sdW_corr = self.sdW / (1-beta2**t)
                vdW_corr /= (np.sqrt(sdW_corr) + epsilon)

                self.sdb = beta2 * self.sdb + (1-beta2) * self.db ** 2
                sdb_corr = self.sdb / (1-beta2**t)
                vdb_corr /= (np.sqrt(sdb_corr) + epsilon)
            
            self.W = self.W - (learning_rate * vdW_corr)
            self.b = self.b - (learning_rate * vdb_corr)

        


    def _linear_fwd(self, A_prev):
        return np.dot(self.W,A_prev) + self.b


    def _linear_bwd(self, dZ, A_prev, lambd):
        m = dZ.shape[1]

        self.dW = 1/m * np.dot(dZ, A_prev.T) + (lambd/m) * self.W
        self.db = 1/m * np.sum(dZ, axis=1, keepdims = True)
        
        
        dA_prev = np.dot(self.W.T, dZ)
        
        return dA_prev

    @staticmethod
    def _sig_fwd(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def _sig_bwd(Z):
        return Layer._sig_fwd(Z) * (1 - Layer._sig_fwd(Z))
        
    @staticmethod
    def _relu_fwd(Z):
        return np.maximum(Z,0)

    @staticmethod
    def _relu_bwd(Z):
        return np.int8(Z>0)

    def __str__(self):
        return '\tW.shape: {}\n\tb.shape: {}\n\tinitialization method: {}\n\tactivation method: {}'.format(
            self.W.shape, self.b.shape, self.init_method, self.act_func)

class NeuralNetwork:

    optimizers = ['gd', 'momentum', 'adam']
    
    def __init__(self, layer_dims, init_method = 'standard'):
        self.L = len(layer_dims) - 1
        self.AL = None
        self.layers = {}
        for l in range(1, self.L + 1):
            act_func = 'relu'
            if l == self.L:
                act_func = 'sigmoid'
            self.layers[l] = Layer(layer_dims[l],layer_dims[l-1], act_func, init_method = init_method)


    def train(self, train_X, train_Y, learning_rate = 0.05, batch_size = 64, num_epochs = 50, optimizer = 'gd', lambd = 0, keep_prob = 1, beta1 = 0.9, beta2 = 0.999, epsilon = 10**-8, print_costs = True):
        m = train_X.shape[1]
        
        assert m == train_Y.shape[1]
        assert optimizer in self.optimizers

        num_batches = math.ceil(m/batch_size)

        #counter that keeps track of the number of total times optimizer has been run (for use with Adam)
        t = 1
        costs = []
        for i in range(num_epochs):
            #shuffle training data
            shuffled = list(np.random.permutation(m))
            shuffled_X = train_X[:,shuffled]
            shuffled_Y = train_Y[:,shuffled].reshape(train_Y.shape)

            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = (batch+1) * batch_size
                minibatch_X = shuffled_X[:,start_idx:end_idx]
                minibatch_Y = shuffled_Y[:,start_idx:end_idx]

                self._forward_prop(minibatch_X, keep_prob)
                cost = NeuralNetwork.cost(self.AL, minibatch_Y, self.layers, lambd)
                self._backward_prop(minibatch_X, minibatch_Y, keep_prob, lambd)
                self._update_params(learning_rate, optimizer, beta1, beta2, epsilon, t)
                
                t += 1
                
            if i % 100 == 0:
                costs.append(cost)
            if print_costs and i % 1000 == 0:
                print('Cost after epoch {}/{}: {}'.format(i, num_epochs, cost))
        return costs

    def predict(self, X):
        self._forward_prop(X, keep_prob = 1)    #don't do dropout when predicting
        y_hat = np.copy(self.AL)

        #binary classification
        if y_hat.shape[0] == 1:
            y_hat[y_hat > 0.5] = 1
            y_hat[y_hat <= 0.5] = 0

        #multiclass classification
        else:
            y_hat = np.argmax(y_hat, axis=0)

        return y_hat

    @staticmethod
    def cost(AL, Y, layers, lambd = 0):
        assert AL.shape == Y.shape

        m = Y.shape[1]

        #Prevents division by zero
        AL[AL == 0] = 0.000001
        AL[AL == 1] = 0.999999

        cost = -1/m * np.sum((Y * np.log(AL) + (1-Y) * np.log(1-AL)))
        cost = np.squeeze(cost)
        assert cost.shape == ()

        #L2 regularization
        if lambd != 0:
            l2_reg_cost = 0

            for l in layers:
                l2_reg_cost += np.sum(layers[l].W ** 2)
            l2_reg_cost = (lambd / (2 *m)) * l2_reg_cost 
            cost += l2_reg_cost
        return cost

    @staticmethod
    def print_accuracy(Y_hat, Y, dataset_name='Overall'):
        acc = NeuralNetwork.accuracy(Y_hat,Y)
        print('{} accuracy: {}%'.format(dataset_name, acc * 100))

    @staticmethod
    def accuracy(Y_hat, Y):
        num_classes, m = Y.shape

        mult = np.array(range(1, num_classes+1)).reshape(num_classes,1)
        Y_hat = Y_hat * mult
        Y = Y * mult
        Y_hat = np.sum(Y_hat, axis = 0)
        Y = np.sum(Y, axis = 0)
        correct = Y_hat == Y
        acc = (np.sum(correct) / m)
        return acc

    def _forward_prop(self, X, keep_prob):
        A = X
        for l in self.layers:
            curr_layer = self.layers[l]
            if l == self.L:
                keep_prob = 1 #no dropout in last layer
            A = curr_layer.forward_prop(A, keep_prob)
        self.AL = A

    def _backward_prop(self, X, Y, keep_prob, lambd):

        #Prevents division by zero
        self.AL[self.AL == 0] = 0.000001
        self.AL[self.AL == 1] = 0.999999

        dA = -1 * (Y/self.AL - (1-Y)/(1-self.AL))
        for l in reversed(range(1,self.L + 1)):
            if l != 1:
                A_prev = self.layers[l-1].A
            elif l == 1:
                A_prev = X
            curr_layer = self.layers[l]
            
            #no dropout on last layer
            if l == self.L:
                dA = curr_layer.backward_prop(dA, A_prev, 1, lambd)
            else:
                dA = curr_layer.backward_prop(dA, A_prev, keep_prob, lambd)

    def _update_params(self, learning_rate, optimizer, beta1, beta2, epsilon, t):
        for l in self.layers:
            self.layers[l].update_params(learning_rate, optimizer, beta1, beta2, epsilon, t)

    def __str__(self):
        to_return = ''
        for l in self.layers:
            to_return += 'Layer {}:\n{}\n\n'.format(l,self.layers[l])
        return to_return

import matplotlib.pyplot as plt


if __name__ == '__main__':

    nn = NeuralNetwork([1000,200,20,10,5], init_method = 'he')
    print(nn)

