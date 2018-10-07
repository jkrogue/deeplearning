'''
deep_neural_network.py
This module provides implementation of a neural network via nested classes.
The neural network class consists of multiple Layer objects, each of which
    contains parameters pertaining to that layer (e.g., W, b, activation function, etc).

Author = Justin Krogue
'''
import numpy as np

class Layer:
    """
    Represents a single layer in a neural network
    
    Stored parameters:
        W = matrix of weights
        b = vector of biases
        activation = name of activation function for this layer (either 'relu' or 'sigmoid')
        Z, A = matrices of outputs, initialized to None and are updated whenever forward_prop
            is run
        dW, db = partial derivatives of loss with respec to W and b of this layer.
            Initialized to None, updated whenever backward_prop is run
            
    To use, first initialize the layer with appropriate parameters, then run forward_prop,
        backward_prop, and update_params
        
    All private functions are only needed for internal use
    """
    
    # list of acceptable activations
    activations = ['relu','sigmoid']
    initializations = ['he','xavier','standard']
    
    def __init__(self, n_curr, n_prev, activation, initialization='standard', lambd = 0, keep_prob = 1):
        """
        Arguments:
        n_curr = number of nodes in this layer
        n_prev = number of nodes in previous layer
        prev_layer = 
        activation = name of activation function, must be either 'relu' or 'sigmoid'
        initialization = which method to use to scale weights, must be either 'standard', 
            'he', or 'xavier'
        lambd = lambda value used for L2 regularization (will do nothing if = 0)
        keep_prob = probability of node being included in dropout (all included if = 1)
        """
        print(initialization)
        if initialization not in self.initializations:
            raise Exception('invalid initilization method')

        if initialization == 'he':
            self.W = np.random.randn(n_curr,n_prev) * np.sqrt(2/n_prev)
        elif initialization == 'xavier':
            self.W = np.random.randn(n_curr,n_prev) / np.sqrt(n_prev)
        else:
            self.W = np.random.randn(n_curr,n_prev) * 0.01
        
        self.b = np.zeros((n_curr,1))
        self.Z = None
        self.A = None
        self.dW = None
        self.db = None
        self.D = None
        self.lambd = lambd
        self.keep_prob = keep_prob
        
        if activation not in self.activations:
            raise Exception('invalid activation function name')
        else:
            self.activation = activation
        
        assert (self.W.shape == (n_curr, n_prev))
        assert(self.b.shape == (n_curr, 1))
        
    def forward_prop(self, A_prev, dropout = False):
        """
        Calculate Z and A.  These are stored internally and then A is returned
        
        Arguments:
        A_prev = output of layer before this one (e.g., output of layer 2 if this is layer 3)
        dropout = boolean indicating whether you want dropout performed
        
        Returns:
        A
        """
        
        self.Z = self._linear_fwd(A_prev)
        self.A = self._act_fwd(self.Z)
        
        if dropout and self.keep_prob < 1:
            self.D = np.random.rand(self.A.shape[0],self.A.shape[1])
            self.D = self.D < self.keep_prob
            self.A *= self.D
            self.A /= self.keep_prob
                                    
        return self.A

    def backward_prop(self, dA, A_prev):
        """
        Calculate partial derivates dW, db, and dA_prev (partial derivative with respect to
            output of previous layer). dW and db are stored internally, dA_prev is returned
        
        Arguments:
        dA = partial derivative of loss with respect to activation output of this layer
        A_prev = output of layer before this one (e.g., output of layer 2 if this is layer 3)
        
        Returns:
        dA_prev (partial derivative with respect to output of previous layer)
        """
        
        dZ = self._act_bwd(dA)
        self.dW, self.db, dA_prev = self._linear_bwd(dZ, A_prev)
                
        return dA_prev
    
    def update_params(self, learning_rate):
        """
        Updates parameters (W and b) in this layer according to learning_rate specified and
            stored values of W, b, dW, and db
        
        Arguments:
        learning_rate
        
        Returns: none (updated parameters are stored internally)
        """
        self.W = self.W - (learning_rate * self.dW)
        self.b = self.b - (learning_rate * self.db)
        
    def _linear_fwd(self, A_prev):
        """
        Private method for internal use only.
        Calculates Z using this layer's parameters (W, b) and output of previous layer
        
        Arguments:
        A_prev = output of previous layer (e.g., of layer 2 if this is layer 3)
        
        Returns:
        Z
        """
        
        Z = np.dot(self.W,A_prev) + self.b
        assert(Z.shape == (self.W.shape[0],A_prev.shape[1]))
        return Z
    
    def _act_fwd(self, Z):
        """
        Private method for internal use only.
        Calculates final output of this layer using appropriate activation function and
            supplied linear output Z
        
        Arguments:
        Z = linear output of this layer
        
        Returns:
        A (final output of layer)
        """
        
        if self.activation == 'relu':
            A = Layer._relu(self.Z)
        elif self.activation == 'sigmoid':
            A = Layer._sigmoid(self.Z)
            
        assert(A.shape == Z.shape)
        return A
    
    def _linear_bwd(self, dZ, A_prev):
        """
        Private method for internal use only.
        
        Calculates partial derivatives of loss with respect to this layer's parameters (dW, db)
            and previous layer's output (dA_prev) given partial derivative of loss with respect
            to linear output of this layer (dZ), and final output of previous layer (A_prev)
        
        Arguments:
        dZ = partial derivative of loss with respect to linear output of this layer
        A_prev = output of previous layer (e.g., of layer 2 if this is layer 3)
        
        Returns:
        dW, db, dA_prev
        """
        
        m = dZ.shape[1]

        dW = 1/m * np.dot(dZ, A_prev.T) + ((self.lambd/m) * self.W)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)
        
        assert dW.shape == self.W.shape
        assert db.shape == self.b.shape
        assert dA_prev.shape == A_prev.shape
        
        return dW, db, dA_prev

        
    def _act_bwd(self, dA):
        """
        Private method for internal use only.
        
        Calculates partial derivatives of loss with respect to this layer's linear output (dZ),
            given partial derivative of loss with respect to final output (dA)
        
        Arguments:
        dA = partial derivative of loss with respect to final output of this layer
        
        Returns:
        dZ
        """

        # perform dropout if keep_prob < 1
        if self.keep_prob < 1:
            dA = self.D * dA
            dA = dA / self.keep_prob

        if self.activation == 'relu':
            dZ = Layer._relu_bwd(dA, self.Z)
        elif self.activation == 'sigmoid':
            dZ = Layer._sigmoid_bwd(dA, self.Z)
            
        assert (dZ.shape == dA.shape == self.Z.shape)
        
        return dZ

    @staticmethod
    def _sigmoid(Z):
        """
        Private, static method for internal use only (call with Layer._sigmoid)
        
        Arguments:
        Z = linear output of layer
        
        Returns:
        sigmoid function of Z
        """

        return 1 / (1+np.exp(-Z))
    
    @staticmethod
    def _sigmoid_bwd(dA, Z):
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

        s = Layer._sigmoid(Z)
        dZ = dA * s * (1-s)
        return dZ

    @staticmethod
    def _relu(Z):
        """
        Private, static method for internal use only (call with Layer._rele)
        
        Arguments:
        Z = linear output of layer
        
        Returns:
        relu function of Z
        """

        return np.maximum(0,Z)
    
    @staticmethod
    def _relu_bwd(dA, Z):
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

        dZ = np.array(dA, copy=True) # just converting dz to a correct object.

        dZ = np.multiply(dZ, np.int64(Z > 0))

        return dZ


class neural_network:
    """
    Implementation of extensible neural network
    
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
    
    def __init__(self, layer_dims, initialization = 'standard', lambd = 0, keep_prob = 1):
        """
        Initialize network
        
        Arguments:
        layer_dims: list of node sizes of network (e.g., [5,4,1] represents 2 layer network
            with 5 inputs, 4 nodes in hidden layer, and 1 node in output layer)
            
        initialization = which method to use to scale weights, must be either 'standard',
            'he', or 'xavier'
            
        lambd = lambda value used for L2 regularization (will do nothing if = 0)
        keep_prob = probability of node being included in dropout (all included if = 1)

        """
        self.L = len(layer_dims) - 1
        self.layers = {}
        self.AL = None
        
        for l in range(1,self.L+1):
            activation = 'relu'
            if l == (self.L): #last layer, use sigmoid activation function
                activation = 'sigmoid'
                keep_prob = 1 #no dropout in the last layer

            curr_layer = Layer(layer_dims[l],layer_dims[l-1],activation, initialization = initialization,
                               lambd = lambd, keep_prob = keep_prob)
            assert(curr_layer.W.shape == (layer_dims[l],layer_dims[l-1]))
            assert(curr_layer.b.shape == (layer_dims[l],1))
            self.layers[l] = curr_layer

        assert(len(self.layers) == self.L)

    def train(self, X, Y, learning_rate = 0.05, num_iterations = 2000, print_costs = True, print_interval = 100):
        """
        Trains the network by calling _forward_prop and _backward_prop repeatedly.
        
        Arguments
        X = matrix of inputs
        Y = matrix of correct answers
        learning_rate
        num_iterations = number of times to update parameters
        print_costs = if true, then will print out cost every 100 iterations
        
        Returns
        costs: a list of the costs after every 100 iterations
        """
                
        assert(X.shape[1] == Y.shape[1])
        assert(X.shape[0] == self.layers[1].W.shape[1] and Y.shape[0] == self.layers[self.L].W.shape[0])
        
        costs = []
        for i in range(num_iterations):
            #np.random.seed(1) uncomment to run with regularization exercise
            self._forward_prop(X, dropout=True)
            self._backward_prop(X, Y, learning_rate)
            
            cost_interval = print_interval / 10
            if print_costs and i % cost_interval == 0:
                cost = self.cost(Y)
                costs.append(cost)
            # Print the loss every 10000 iterations
            if print_costs and i % print_interval == 0:
                print("Cost after iteration {}: {}".format(i, cost))

        return costs
                    
    def predict(self, X):
        """
        Produces binarized predictions (0 or 1) given current final output of network
        
        Returns
        Binarized predictions
        """
        self._forward_prop(X, dropout=False)
        Y_hat = np.array(self.AL, copy=True)
        Y_hat[Y_hat > 0.5] = 1
        Y_hat[Y_hat <= 0.5] = 0
        
        return Y_hat
        
    def cost(self, Y):
        """
        Returns cost of network given current output and Y
        
        Arguments:
        Y = correct answers
        
        Returns:
        cost
        """
        
        m = Y.shape[1]
        cost = - 1/m *  np.sum(Y*np.log(self.AL) + (1-Y)*(np.log(1-self.AL)))
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        
        l2_cost = 0
        for l in self.layers:
            l2_cost += np.sum(self.layers[l].W ** 2)
        l2_cost *= (self.layers[1].lambd / (2*m))
        cost += l2_cost
        
        return cost
        
    @staticmethod
    def accuracy(y_hat, y):
        """
        Calculate accuracy of predictions (percent correct)
        
        Arguments:
        y_hat: predictions
        y: ground truth
        
        Returns
        accuracy = percent of predictions that were correct from 0-100%
        """
        
        correct = y_hat == y
        true_predictions = correct[correct==True]
        true_predictions = true_predictions.reshape(1,true_predictions.shape[0])
        accuracy = (true_predictions.shape[1] / y.shape[1]) * 100
        return accuracy
    
    @staticmethod
    def print_accuracy(y_hat, y, dataset_name):
        """
        Method that prints out accuracy of predictions from 0-100%
        
        Arguments:
        y_hat: predictions
        y: ground truth
        dataset_name: name of the dataset being graded (e.g., 'test' or 'train')
        """
        percent_right = neural_network.accuracy(y_hat, y)
        print("{} accuracy: {:.2f}%".format(dataset_name,percent_right))

    def _forward_prop(self,X, dropout=False):
        """
        Private method for internal use.
        
        Implements forward propagation through each layer of network, and updates final output
        
        Arguments
        X = inputs
        dropout = whether or not dropout should be performed
        
        Returns
        None (AL is updated internally)
        """

        m = X.shape[1]
        assert(X.shape[0] == self.layers[1].W.shape[1])
        
        A_prev = X
        
        for l in self.layers:
            curr_layer = self.layers[l]
            A_prev = curr_layer.forward_prop(A_prev, dropout = dropout)
        
        self.AL = A_prev
    
    def _backward_prop(self,X,Y,learning_rate):
        """
        Private method for internal use.
        
        Implements backward propagation and updates parameters through each layer of network
        
        Arguments
        X = inputs
        Y = correct answers
        learning_rate
        
        Returns
        None (layer parameters are updated internally)
        """

        #Prevents division by zero
        self.AL[self.AL == 0] = 0.000001
        self.AL[self.AL == 1] = 0.999999

        dA = -1 * (Y/self.AL - (1-Y)/(1-self.AL))
        
        for l in reversed(range(1,self.L+1)):
            if l != 1:
                A_prev = self.layers[l-1].A
            else:
                A_prev = X
            
            curr_layer = self.layers[l]

            dA = curr_layer.backward_prop(dA, A_prev)
            curr_layer.update_params(learning_rate)
        
        
    def __str__(self):
        """
        Provides string representation of neural network.  Prints out shape and activation 
        function of each layer
        """
        
        to_return = ''
        for l in self.layers:
            layer = self.layers[l]
            to_return += "Layer: {}\n\tW.shape = {}\n\tb.shape = {}\n\tactivation function = {}\n\tkeep_prob: {}\n\tlambda: {}\n\n".format(l,layer.W.shape,layer.b.shape,layer.activation,layer.keep_prob,layer.lambd)
        return to_return


if __name__ == '__main__':
    layer = Layer(3,2, 'sigmoid')

    A_prev = np.random.randn(2,500) * 0.01

    A = layer.forward_prop(A_prev)
    print(A.shape)
    A

    layer = Layer(3,2, 'relu')

    A = layer.forward_prop(A_prev)
    print(A.shape)
    A

    np.random.seed(1)

    X = np.random.randint(1,4,(3,2000))
    print(X.shape)
    Y = np.sum(X, axis=0, keepdims=True) > 5
    print(Y.shape)

    nn = neural_network([3,5,1])
    print(nn)

    costs = nn.train(X,Y,num_iterations=2000,learning_rate = 0.05)
    neural_network.print_accuracy(nn.predict(X),Y,'Overall')