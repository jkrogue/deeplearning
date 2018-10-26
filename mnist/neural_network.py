import numpy as np
import math

'''
neural_network.py
This module provides implementation of a densely-connected neural network via nested classes.
The neural network class consists of multiple Layer objects, each of which
    contains parameters pertaining to that layer (e.g., W, b, activation function, etc).

Author = Justin Krogue
'''
class Layer:

    """
    Represents a single layer in a neural network
    
    Stored parameters:
        W = matrix of weights
        b = vector of biases
        act_func = name of activation function for this layer (either 'relu' or 'sigmoid')
        Z, A = matrices of outputs, initialized to None and are updated whenever forward_prop
            is run
        dW, db = partial derivatives of loss with respec to W and b of this layer.
            Initialized to None, updated whenever backward_prop is run
        D = matrix of numbers from 0-1 matching dimensions of layer's output to be used in dropout
        vdW = exponentially-weighted average of dW to be used in momemtum optimization
        sdW = exponentially-weighted average of square of dW to be used to implement RMS-prop portion
            of Adam optimization
            
    To use, first initialize the layer with appropriate parameters, then run forward_prop,
        backward_prop, and update_params
        
    All private functions are only needed for internal use
    """

    #list of acceptable activations
    act_funcs= ['relu', 'sigmoid', 'softmax']

    #list of acceptable initialization methods
    init_methods = ['standard', 'xavier', 'he']

    def __init__(self, n_curr, n_prev, act_func, init_method = 'standard'):
        """
        Arguments:
        n_curr = number of nodes in this layer
        n_prev = number of nodes in previous layer
        act_func = name of activation function, must be either 'relu' or 'sigmoid'
        init_method = which method to use to scale weights, must be either 'standard', 
            'he', or 'xavier'
        """

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
        elif act_func == 'softmax':
            self._act_fwd = Layer._softmax_fwd


    def forward_prop(self, A_prev, keep_prob):
        """
        Calculate Z and A.  These are stored internally and then A is returned
        
        Arguments:
        A_prev = output of layer before this one (e.g., output of layer 2 if this is layer 3)
        keep_prob = used for implementing dropout.  Float from 0-1 indicating what percent of 
            neurons output to keep.  E.g., keep_prob = 1: normal forward prop including all
            outputs.  keep_prob = 0.5: 50% dropout
        Returns:
        A
        """

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
        """
        Calculate partial derivates dW, db, and dA_prev (partial derivative with respect to
            output of previous layer). dW and db are stored internally, dA_prev is returned
        
        Arguments:
        dA = partial derivative of loss with respect to activation output of this layer
        A_prev = output of layer before this one (e.g., output of layer 2 if this is layer 3)
        keep_prob = used for implementing dropout.  Float from 0-1 indicating what percent of 
            neurons output to keep.  E.g., keep_prob = 1: normal forward prop including all
            outputs.  keep_prob = 0.5: 50% dropout
        
        Returns:
        dA_prev (partial derivative with respect to output of previous layer)
        """
        assert (self.A.shape == dA.shape)

        if keep_prob < 1:
            dA *= self.D
            dA /= keep_prob

        dZ = np.multiply(dA,self._act_bwd(self.Z))
        dA_prev = self._linear_bwd(dZ, A_prev, lambd)

        assert(dA_prev.shape == (self.dW.shape[1],dA.shape[1]))

        return dA_prev

    def update_params(self, learning_rate, optimizer, beta1, beta2, epsilon, t):
        """
        Updates parameters (W and b) in this layer according to learning_rate specified and
            stored values of W, b, dW, and db
        
        Arguments:
        learning_rate
        optimizer = name of optimization algorithms to use (either 'gd' for gradient descent, 
            'momentum,' or 'Adam')
        beta1 = beta factor for use in momentum
        beta2 = beta factor for use in RMS prop
        epsilon = small factor to prevent division by 0 in Adam algorithm
        t = current iteration number, to be used in momentum and Adam algorithms 
        
        Returns: none (updated parameters are stored internally)
        """

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
        """
        Private method for internal use only.
        Calculates Z using this layer's parameters (W, b) and output of previous layer
        
        Arguments:
        A_prev = output of previous layer (e.g., of layer 2 if this is layer 3)
        
        Returns:
        Z
        """

        return np.dot(self.W,A_prev) + self.b


    def _linear_bwd(self, dZ, A_prev, lambd):
        """
        Private method for internal use only.
        
        Calculates partial derivatives of loss with respect to this layer's parameters (dW, db)
            and previous layer's output (dA_prev) given partial derivative of loss with respect
            to linear output of this layer (dZ), and final output of previous layer (A_prev).

        dW and db are updated internally, dA_prev is returned
        
        Arguments:
        dZ = partial derivative of loss with respect to linear output of this layer
        A_prev = output of previous layer (e.g., of layer 2 if this is layer 3)
        lambd = penalization-factor to use in L2 regularization 
        
        Returns:
        dA_prev
        """

        m = dZ.shape[1]

        self.dW = 1/m * np.dot(dZ, A_prev.T) + (lambd/m) * self.W
        self.db = 1/m * np.sum(dZ, axis=1, keepdims = True)
        
        
        dA_prev = np.dot(self.W.T, dZ)
        
        return dA_prev

    @staticmethod
    def _sig_fwd(Z):
        """
        Private, static method for internal use only (call with Layer._sigmoid)
        
        Arguments:
        Z = linear output of layer
        
        Returns:
        sigmoid function of Z
        """

        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def _sig_bwd(Z):
        """
        Private, static method for internal use only (call with Layer._sigmoid_bwd).
        
        Calculates and returns partial derivative of loss with respect to linear output 
            of layer by taking derivative of sigmoid function and multiplying by partial 
            derivative of loss with regards to final output of layer (chain rule).
        
        Arguments:
        dA = partial derivative of loss with respect to final output of layer
        Z = linear output of layer
        
        Returns:
        dZ = partial derivative of loss with respect to linear output of layer
        """

        return Layer._sig_fwd(Z) * (1 - Layer._sig_fwd(Z))
        
    @staticmethod
    def _relu_fwd(Z):
        """
        Private, static method for internal use only (call with Layer._rele)
        
        Arguments:
        Z = linear output of layer
        
        Returns:
        relu function of Z
        """

        return np.maximum(Z,0)

    @staticmethod
    def _relu_bwd(Z):
        """
        Private, static method for internal use only (call with Layer._relu_bwd).
        
        Calculates and returns partial derivative of loss with respect to linear output 
            of layer by taking derivative of relu function and multiplying by partial 
            derivative of loss with regards to final output of layer (chain rule).
        
        Arguments:
        dA = partial derivative of loss with respect to final output of layer
        Z = linear output of layer
        
        Returns:
        dZ = partial derivative of loss with respect to linear output of layer
        """

        return np.int8(Z>0)

    @staticmethod
    def _softmax_fwd(Z):
        """
        Private, static method for internal use only (call with Layer._sigmoid)
        
        Arguments:
        Z = linear output of layer
        
        Returns:
        softmax activation function of Z (first computes e^Z elementwise and then
            divides by sum of all e^Z)
        """

        T = np.exp(Z)

        sum_T = np.sum(T, axis=0)
        A = T / sum_T
        assert(A.shape == Z.shape)
        assert(np.sum(A[:,0]) >= 0.999), np.sum(A[:,0])
        return A

    def __str__(self):
        '''
        Returns string representation of this layer containing shape of W and b, initialiation method used,
            and activation function used
        '''

        return '\tW.shape: {}\n\tb.shape: {}\n\tinitialization method: {}\n\tactivation method: {}'.format(
            self.W.shape, self.b.shape, self.init_method, self.act_func)

class NeuralNetwork:
    """
    Implementation of densely-connected neural network, including options to allow differing initialization
        methods (e.g., He or Xavier), regularization methods (L2 regularization, dropout), and optimization
        algorithms (mini-batch, momentum, Adam)
    
    Stored parameters:
        layers = dictionary of Layer objects where key is equal to layer number
            (i.e., self.layers[1] is first layer of network)
        L = number of layers
        AL = matrix of final output of network, initialized to zero initially.
            Updated whenever forward_prop is called
    
    To use, first initialize, and then run train.  Call predict to get binarized predictions,
        and use static methods accuracy and print_accuracy to generate accuracies
    
    All private methods are for internal use and shouldn't need to be called.
    """

    #Acceptable list of optimization algorithms
    optimizers = ['gd', 'momentum', 'adam']
    
    def __init__(self, layer_dims, init_method = 'standard'):
        """
        Initialize network
        
        Arguments:
        layer_dims: list of node sizes of network (e.g., [5,4,1] represents 2 layer network
            with 5 inputs, 4 nodes in hidden layer, and 1 node in output layer)
            
        init_method = which method to use to scale weights, must be either 'standard',
            'he', or 'xavier'
        """

        self.L = len(layer_dims) - 1
        self.AL = None
        self.layers = {}
        for l in range(1, self.L + 1):
            act_func = 'relu'
            if l == self.L:
                act_func = 'sigmoid'
                if layer_dims[l] > 1:
                    act_func = 'softmax'
            self.layers[l] = Layer(layer_dims[l],layer_dims[l-1], act_func, init_method = init_method)


    def train(self, train_X, train_Y, learning_rate = 0.05, batch_size = 64, num_epochs = 50, optimizer = 'gd', 
        lambd = 0, keep_prob = 1, beta1 = 0.9, beta2 = 0.999, epsilon = 10**-8, print_int = 1, print_costs = True):
        """
        Trains the network by calling _forward_prop, _backward_prop and _update_params repeatedly.
        
        Arguments
        X = matrix of inputs
        Y = matrix of correct answers
        learning_rate
        batch_size = size of batch to be used (if want to batch gradient descent, make this equal to number of 
            training examples)
        num_epochs = number of times to run through training data
        optimizer = optimization algorithm to use, either 'gd' (gradient descent), 'momentum', 'Adam'
        lambd = penalization parameter used for L2 regularization (will do nothing if = 0)
        keep_prob = probability of node being included in dropout (all included if = 1)
        beta1 = beta parameter to calculate exponentially-weighted average in Momentum
        beta2 = beta parameter to calculate exponentially-weighted average in RMS-prop (portion of Adam)
        epsilon = small number used to prevent division by zero in Adam optimizer
        print_int = number of epochs after which to print cost
        print_costs = if true, then will print out cost
        
        Returns
        costs: a list of the costs after every epoch
        """

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
                
            costs.append(cost)
            if print_costs and i % print_int == 0:
                print('Cost after epoch {}/{}: {}'.format(i, num_epochs, cost))
        return costs

    def predict(self, X):
        """
        Produces binarized predictions (0 or 1) given current final output of network
        
        Returns
        Binarized predictions
        """

        self._forward_prop(X, keep_prob = 1)    #don't do dropout when predicting
        y_hat = np.copy(self.AL)

        #binary classification
        if y_hat.shape[0] == 1:
            y_hat[y_hat > 0.5] = 1
            y_hat[y_hat <= 0.5] = 0

        #multiclass classification
        else:
            y_hat[y_hat == np.amax(y_hat, axis = 0)] = 1
            y_hat[y_hat < 1] = 0

        return y_hat

    @staticmethod
    def cost(AL, Y, layers, lambd = 0):
        """
        Static method that returns cost given specified model output, and actual labels.
            Allows for L2 regularization via specification of lambda parameter
        
        Arguments:
        AL = model's output
        Y = correct answers
        lambd = penalization parameter used for L2 regularization (will do nothing if = 0)

        Returns:
        cost
        """

        assert AL.shape == Y.shape

        n = Y.shape[0]
        m = Y.shape[1]

        #Prevents division by zero
        AL[AL == 0] = 0.000001
        AL[AL == 1] = 0.999999

        # binary classification
        if n == 1:
            cost = -1/m * np.sum((Y * np.log(AL) + (1-Y) * np.log(1-AL)))
            cost = np.squeeze(cost)
            assert cost.shape == ()
        
        # multiclass classifier, use loss function for softmax
        else:
            loss_matrix = -Y * np.log(AL)
            loss_vector = np.sum(loss_matrix, axis = 0, keepdims=True)
            assert(loss_vector.shape == (1,m))
            cost = np.squeeze(np.sum(loss_vector))

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
        """
        Method that prints out accuracy of predictions from 0-100%
        
        Arguments:
        y_hat: predictions
        y: ground truth
        dataset_name: name of the dataset being graded (e.g., 'test' or 'train')
        """

        acc = NeuralNetwork.accuracy(Y_hat,Y)
        print('{} accuracy: {}%'.format(dataset_name, acc * 100))

    @staticmethod
    def accuracy(Y_hat, Y):
        """
        Calculate accuracy of predictions (percent correct)
        
        Arguments:
        y_hat: predictions
        y: ground truth
        
        Returns
        accuracy = fraction of predictions that were correct from 0-1
        """
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
        """
        Private method for internal use.
        
        Implements forward propagation through each layer of network, and updates final output
        
        Arguments
        X = inputs
        keep_prob = used for implementing dropout.  Float from 0-1 indicating what percent of 
            neurons output to keep.  E.g., keep_prob = 1: normal forward prop including all
            outputs.  keep_prob = 0.5: 50% dropout
        
        Returns
        None (AL is updated internally)
        """

        A = X
        for l in self.layers:
            curr_layer = self.layers[l]
            if l == self.L:
                keep_prob = 1 #no dropout in last layer
            A = curr_layer.forward_prop(A, keep_prob)
        self.AL = A

    def _backward_prop(self, X, Y, keep_prob, lambd):
        """
        Private method for internal use.
        
        Implements backward propagation and updates parameters through each layer of network
        
        Arguments
        X = inputs
        Y = correct answers
        keep_prob = used for implementing dropout.  Float from 0-1 indicating what percent of 
            neurons output to keep.  E.g., keep_prob = 1: normal forward prop including all
            outputs.  keep_prob = 0.5: 50% dropout
        lambd = penalization parameter used for L2 regularization (will do nothing if = 0)

        Returns
        None (layer parameters are updated internally)
        """

        softmax = Y.shape[1] > 1

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
            
            if l == self.L:

                # implement backprop for softmax layer
                if softmax:
                    curr_layer.dZ = self.AL - Y
                    dA = curr_layer._linear_bwd(curr_layer.dZ, A_prev, lambd)

                # no dropout on last layer
                else:
                    dA = curr_layer.backward_prop(dA, A_prev, 1, lambd)

            else:
                dA = curr_layer.backward_prop(dA, A_prev, keep_prob, lambd)

    def _update_params(self, learning_rate, optimizer, beta1, beta2, epsilon, t):
        '''
        Runs through each layer of network and updates the parameters appropriately

        Arguments
        learning_rate
        optimizer = optimization algorithm to use, either 'gd' (gradient descent), 'momentum', 'Adam'
        beta1 = beta parameter to calculate exponentially-weighted average in Momentum
        beta2 = beta parameter to calculate exponentially-weighted average in RMS-prop (portion of Adam)
        epsilon = small number used to prevent division by zero in Adam optimizer
        t = current iteration number to be used for momentum and Adam
        
        Returns
        None (parameters stored internally)
        '''

        for l in self.layers:
            self.layers[l].update_params(learning_rate, optimizer, beta1, beta2, epsilon, t)

    def __str__(self):
        '''
        Provides string representation of neural network layer by layer

        '''
        to_return = ''
        for l in self.layers:
            to_return += 'Layer {}:\n{}\n\n'.format(l,self.layers[l])
        return to_return




if __name__ == '__main__':
    import datetime
    import matplotlib.pyplot as plt

    layer = Layer(3,2, 'sigmoid')

    A_prev = np.random.randn(2,500) * 0.01

    A = layer.forward_prop(A_prev, keep_prob = 1)
    print(A.shape)
    A

    layer = Layer(3,2, 'relu')

    A = layer.forward_prop(A_prev, keep_prob = 1)
    print(A.shape)
    A

    np.random.seed(1)

    X = np.random.randint(1,4,(3,2000))
    print(X.shape)
    Y = np.sum(X, axis=0, keepdims=True) > 5
    X = X / np.max(X)   #normalize
    print(Y.shape)

    #Ensure no errors occur with different initialization methods
    nn = NeuralNetwork([3,5,1], init_method = 'standard')
    nn = NeuralNetwork([3,5,1], init_method = 'xavier')
    nn = NeuralNetwork([3,5,1], init_method = 'he')
    print(nn)

    print('gd optimizer')
    np.random.seed(1)
    nn = NeuralNetwork([3,5,1])
    start = datetime.datetime.now()
    costs = nn.train(X,Y,batch_size=X.shape[1],num_epochs = 2000, learning_rate = 0.05, print_int=100)
    elapsed = datetime.datetime.now() - start
    NeuralNetwork.print_accuracy(nn.predict(X),Y,'Training set')
    print("Time elapsed: {}\n".format(elapsed))

    print('momentum optimizer')
    np.random.seed(1)
    nn = NeuralNetwork([3,5,1])
    start = datetime.datetime.now()
    costs = nn.train(X,Y,optimizer = 'momentum',batch_size=X.shape[1],num_epochs = 2000, 
        learning_rate = 0.05, print_int=100)
    elapsed = datetime.datetime.now() - start
    NeuralNetwork.print_accuracy(nn.predict(X),Y,'Training set')
    print("Time elapsed: {}\n".format(elapsed))

    print('adam optimizer')
    np.random.seed(1)
    nn = NeuralNetwork([3,5,1])
    start = datetime.datetime.now()
    costs = nn.train(X,Y,optimizer = 'adam',batch_size=X.shape[1],num_epochs = 2000, 
        learning_rate = 0.05, print_int=100)
    elapsed = datetime.datetime.now() - start
    NeuralNetwork.print_accuracy(nn.predict(X),Y,'Training set')
    print("Time elapsed: {}\n".format(elapsed))

    print('adam optimizer with he initialization')
    np.random.seed(1)
    nn = NeuralNetwork([3,5,1], init_method='he')
    start = datetime.datetime.now()
    costs = nn.train(X,Y,optimizer = 'adam',batch_size=X.shape[1],num_epochs = 2000, 
        learning_rate = 0.05, print_int=100)
    elapsed = datetime.datetime.now() - start
    NeuralNetwork.print_accuracy(nn.predict(X),Y,'Training set')
    print("Time elapsed: {}\n".format(elapsed))

    import math
    batch_size = 64
    m = X.shape[1]
    iter_per_epoch = math.ceil(m/batch_size)
    num_epochs = int(2000/iter_per_epoch)

    print('minibatch with gd optimizer with he initialization')
    np.random.seed(1)
    nn = NeuralNetwork([3,5,1], init_method='he')
    start = datetime.datetime.now()
    costs = nn.train(X,Y,optimizer = 'gd',batch_size=batch_size,num_epochs = num_epochs, 
        learning_rate = 0.05, print_int=5)
    elapsed = datetime.datetime.now() - start
    NeuralNetwork.print_accuracy(nn.predict(X),Y,'Training set')
    print("Time elapsed: {}\n".format(elapsed))

    print('minibatch with momentum optimizer with he initialization')
    np.random.seed(1)
    nn = NeuralNetwork([3,5,1], init_method='he')
    start = datetime.datetime.now()
    costs = nn.train(X,Y,optimizer = 'momentum',batch_size=batch_size,num_epochs = num_epochs, 
        learning_rate = 0.05, print_int=5)
    elapsed = datetime.datetime.now() - start
    NeuralNetwork.print_accuracy(nn.predict(X),Y,'Training set')
    print("Time elapsed: {}\n".format(elapsed))

    print('minibatch with Adam optimizer with he initialization')
    np.random.seed(1)
    nn = NeuralNetwork([3,5,1], init_method='he')
    start = datetime.datetime.now()
    costs = nn.train(X,Y,optimizer = 'adam',batch_size=batch_size,num_epochs = num_epochs, 
        learning_rate = 0.05, print_int=5)
    elapsed = datetime.datetime.now() - start
    NeuralNetwork.print_accuracy(nn.predict(X),Y,'Training set')
    print("Time elapsed: {}\n".format(elapsed))
