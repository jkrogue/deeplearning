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
    
    def __init__(self, n_curr, n_prev, activation):
        """
        Arguments:
        n_curr = number of nodes in this layer
        n_prev = number of nodes in previous layer
        activation = name of activation function, must be either 'relu' or 'sigmoid'
        """
        
        self.W = np.random.randn(n_curr,n_prev) * 0.01
        self.b = np.zeros((n_curr,1))
        self.Z = None
        self.A = None
        self.dW = None
        self.db = None
        
        if activation not in self.activations:
            raise Exception('invalid activation function name')
        else:
            self.activation = activation
        
        assert (self.W.shape == (n_curr, n_prev))
        assert(self.b.shape == (n_curr, 1))
        
    def forward_prop(self, A_prev):
        """
        Calculate Z and A.  These are stored internally and then A is returned
        
        Arguments:
        A_prev = output of layer before this one (e.g., output of layer 2 if this is layer 3)
        
        Returns:
        A
        """
        
        self.Z = self._linear_fwd(A_prev)
        self.A = self._act_fwd(self.Z)
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
        
        dW = 1/m * np.dot(dZ, A_prev.T)
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

        dZ[Z <= 0] = 0
        
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
    
    To use, first initialize, and then run train.  Call predict to get binarized predictions.
    
    All private methods are for internal use and shouldn't need to be called.
    """
    
    def __init__(self, layer_dims):
        """
        Initialize network
        
        Arguments:
        layer_dims: list of node sizes of network (e.g., [5,4,1] represents 2 layer network
            with 5 inputs, 4 nodes in hidden layer, and 1 node in output layer)
        """
        np.random.seed(1)

        self.L = len(layer_dims) - 1
        self.layers = {}
        self.AL = None
        
        for l in range(1,self.L+1):
            activation = 'relu'
            if l == (self.L): #last layer, use sigmoid activation function
                activation = 'sigmoid'

            curr_layer = Layer(layer_dims[l],layer_dims[l-1],activation)
            assert(curr_layer.W.shape == (layer_dims[l],layer_dims[l-1]))
            assert(curr_layer.b.shape == (layer_dims[l],1))
            self.layers[l] = curr_layer

        assert(len(self.layers) == self.L)

    def train(self, X, Y, learning_rate = 0.05, num_iterations = 2000, print_costs = True):
        """
        Trains the network by calling _forward_prop and _backward_prop repeatedly.
        
        Arguments
        X = matrix of inputs
        Y = matrix of correct answers
        learning_rate
        num_iterations = number of times to update parameters
        print_costs = if true, then will print out cost every 100 iterations
        """
        
        assert(X.shape[1] == Y.shape[1])
        assert(X.shape[0] == self.layers[1].W.shape[1] and Y.shape[0] == self.layers[self.L].W.shape[0])
        
        for i in range(num_iterations):
            self._forward_prop(X)
            self._backward_prop(X, Y, learning_rate)
            
            if (i % 100 == 0) and print_costs:
                cost = self.cost(Y)
                print("Cost after {} iterations: {}".format(i,cost))
                if (cost == np.nan):
                    print('yeah!')
                    
    def predict(self):
        """
        Produces binarized predictions (0 or 1) given current final output of network
        
        Returns
        Binarized predictions
        """
        Y_hat = np.array(self.AL, copy=True)
        Y_hat[Y_hat <= 0.5] = 0
        Y_hat[Y_hat > 0.5] = 1
        
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
        
        return cost   

    def _forward_prop(self,X):
        """
        Private method for internal use.
        
        Implements forward propagation through each layer of network, and updates final output
        
        Arguments
        X = inputs
        
        Returns
        None (AL is updated internally)
        """

        m = X.shape[1]
        assert(X.shape[0] == self.layers[1].W.shape[1])
        
        A_prev = X
        
        for l in self.layers:
            curr_layer = self.layers[l]
            A_prev = curr_layer.forward_prop(A_prev)
        
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
            to_return += "Layer: {}\n\tW.shape = {}\n\tb.shape = {}\n\tactivation function = {}\n\n".format(l,layer.W.shape,layer.b.shape,layer.activation)
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

	nn.train(X,Y,num_iterations=2000,learning_rate = 0.05)
	print(nn.AL.shape)
	predictions = nn.predict()
	correct = predictions == Y
	true_predictions = correct[correct == True]
	true_predictions = true_predictions.reshape(1,true_predictions.shape[0])
	print("Accuracy = {}%".format(true_predictions.shape[1]/correct.shape[1] * 100))


