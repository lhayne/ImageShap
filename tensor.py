import numpy as np
from functions import *
from utils import topological_sort
from conv_functions import *

class Tensor(object):
    """
    Tensor. A wrapper around NumPy's vectors
    which supports a backward call.

    Uses operations defined in section 1.1 which takes Tensors
    as arguments and returns a new Tensor with reference to the
    operations's gradient function.
    """
    def __init__(self, value, is_leaf = True, grad_fn = None):
        """
        Constructor

        Every Tensor which is a result of the operations in section
        1.1 is a non leaf node.
        A non leaf node holds reference to the backward() of a
        Function class by which the Tensor has been created. This is
        how the computational graph is built.

        params:

        value (ndarray or scalar): Holds actual data.

        is_leaf (bool): Specifies if the Tensor is a leaf node.

        grad_fn (Function.backward method): The gradient object which holds forward and
            backward calls specific to the operation that results in the creation
            of this Tensor.

        member variables:

        value (ndarray): value in the arguments to the is converted to
            an ndarray to support numpy vectorization.

        """
        if grad_fn is None and not is_leaf:
            raise ValueError(
                'Non leaf nodes require a grad_fn.'
            )

        if np.isscalar(value):
            value = np.ones(1)*value

        if not isinstance(value, np.ndarray):
            raise ValueError(
                'Value should be of type "np.ndarray" or a scalar, but received {type(value)}'
            )

        self.value = value
        self.is_leaf = is_leaf
        self.grad_fn = grad_fn
        self.zero_grad()


    def __repr__(self):
        return 'Tensor(value: {}, grad: {}, grad_fn = {})'.format(
                self.value, self.grad, self.grad_fn
            )

    def zero_grad(self):
        """
        Reset the gradients of this Tensor to 0 taking in consideration
        the dimensions of the data stored by it.
        """
        self.grad = np.zeros(self.value.shape)

    def backward(self, gradient=1.0):
        """
        Initiates the chain rule on the computational graph.
        """
        self.grad = np.ones(self.value.shape)
        graph = topological_sort(self)

        for t in reversed(graph):
            t.grad_fn.backward(t.grad)

    def __add__(self, other):
        if not (isinstance(other, Tensor)):
            raise ValueError('Function arguments need to be an instance of Tensor.')

        """
        Overloaded "+" primitive.
        Example:
        c = a+b
        params:

        self (Tensor): denoted by 'a' in the example expression
        other (Tensor): denoted by 'b' in the example expression

        returns:

        Tensor: denoted by 'c' in the above expression.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        function = Add()
        forward_value = function.forward(self, other)

        return Tensor(
            value = forward_value,
            is_leaf = False,
            grad_fn = function
        )

    def __sub__(self, other):
        """
        Overloaded "-" primitive.
        Example:
        c = a-b
        params:

        self (Tensor): denoted by 'a' in the example expression
        other (Tensor): denoted by 'b' in the example expression

        returns:

        Tensor: denoted by 'c' in the above expression.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)


        function = Sub()

        return Tensor(
                value = function.forward(self, other),
                is_leaf = False,
                grad_fn = function
            )

    def __mul__(self, other):
        """
        Overloaded "*" primitive.
        Example:
        c = a*b
        params:

        self (Tensor): denoted by 'a' in the example expression
        other (Tensor): denoted by 'b' in the example expression

        returns:

        Tensor: denoted by 'c' in the above expression.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        function = Mul()

        return Tensor(
            value = function.forward(self, other),
            is_leaf = False,
            grad_fn = function
        )

    def __truediv__(self, other):

        """
        Overloaded "/" primitive.
        Example:
        c = a/b
        params:

        self (Tensor): denoted by 'a' in the example expression
        other (Tensor): denoted by 'b' in the example expression

        returns:

        Tensor: denoted by 'c' in the above expression.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        function = Div()

        return Tensor(
            value = function.forward(self, other),
            is_leaf = False,
            grad_fn = function
        )

    def sum(self):
        """
        "Sum elements of this Tensor"
        Example:
        b = a.sum()
        params:

        self (Tensor): denoted by 'a' in the example expression

        returns:

        Tensor: denoted by 'b' in the above expression.
        """

        function = Sum()

        return Tensor(
            value = function.forward(self),
            is_leaf = False,
            grad_fn = function
        )

    def relu(self):
        """
        "ReLU activation applied on this Tensor"
        Example:
        b = a.relu()
        params:

        self (Tensor): denoted by 'a' in the example expression

        returns:

        Tensor: denoted by 'b' in the above expression.
        """

        function = ReLU()

        return Tensor(
                value = function.forward(self),
                is_leaf = False,
                grad_fn = function
            )
    
    def leakyrelu(self):
        """
        "Leaky ReLU activation applied on this Tensor"
        Example:
        b = a.relu()
        params:

        self (Tensor): denoted by 'a' in the example expression

        returns:

        Tensor: denoted by 'b' in the above expression.
        """

        function = leakyReLU()

        return Tensor(
                value = function.forward(self),
                is_leaf = False,
                grad_fn = function
            )

    def mean(self):
        """
        "Mean of element of this Tensor"
        Example:
        b = a.mean()
        params:

        self (Tensor): denoted by 'a' in the example expression

        returns:

        Tensor: denoted by 'b' in the above expression.
        """

        function = Mean()

        return Tensor(
                value = function.forward(self),
                is_leaf = False,
                grad_fn = function
            )

    def pow(self, exp):
        """
        "Raise the value in this Tensor to an exponent"
        Example:
        b = a.pow(exp)
        params:

        self (Tensor): denoted by 'a' in the example expression
        exp (int): denoted by 'exp' in the example expression

        returns:

        Tensor: denoted by 'b' in the above expression.
        """

        function = Pow()

        return Tensor(
            value = function.forward(exp, self),
            is_leaf = False,
            grad_fn = function
        )

    def dot(self, other):
        """
        Dot product of 2 Tensors
        Example:
        c = a.dot(b)
        params:

        self (Tensor): denoted by 'a' in the example expression
        other (Tensor): denoted by 'b' in the example expression

        returns:

        Tensor: denoted by 'c' in the above expression.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        function = Dot()

        return Tensor(
                value = function.forward(self, other),
                is_leaf = False,
                grad_fn = function
            )
            
    def abs(self):

        function = Absolute()

        return Tensor(
                value = function.forward(self),
                is_leaf = False,
                grad_fn = function
            )

    def sigmoid(self):

        function = Sigmoid()

        return Tensor(
                value = function.forward(self),
                is_leaf = False,
                grad_fn = function
            )

    def conv2d(self, other, stride=1, padding=0):
        """
        Conv2D operation between two tensors
        Example: a.conv2d(b)
        Note that in_channels of input == in_channels of kernel for conv2d operation.
        :param self : The input tensor of shape (batch_size, in_channels, height, width).
        :param other: The kernel tensor of shape (out_channels, in_channels, height, width).
        :param stride: defaults to 1.
        :param padding: defaults to 0.
        :return:
        """
        function = Convolution2D()

        return Tensor(
                value = function.forward(stride, padding, self, other),
                is_leaf = False,
                grad_fn = function
            )

    def reshape(self, shape):
        """
        Reshape operation on a tensor.
        Example: a.reshape(shape).
        :param shape: tuple containing required dimensions, ex. (1, 1, 2, 3).
        :return: reshaped tensor.
        """
        function = Reshape()

        return Tensor(
                value = function.forward(shape, self),
                is_leaf = False,
                grad_fn = function
            )