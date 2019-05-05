import numpy as np


class Function(object):
    """
    Represents a node in computational graph that perfoms
    a computation.

    During forward mode computation, it takes in
    1 or more inputs/parents, and returns a result of the
    computation as the output.

    In reverse mode accumulation, it takes in the
    gradients w.r.t. the output of the node, accumulates them
    by calculating the gradients w.r.t it's inputs/parents
    and back-propogates the gradients to the parents.
    """
    # List of inputs to the node.
    parents = []

    def __init__(self):
        pass

    def forward(self, *args):
        """
        Forward mode computation of the 'operation'
        on the inputs/parents to be implemented here.
        """
        pass

    def backward(self, gradient):
        """
        Reverse mode computation of the node implemented here.
        """
        pass


class Add(Function):
    """
    Add parent inputs and return the result.
    """
    def forward(self, *args):
        """
        Forward computation of the binary Add operation.

        param:
        args (n=2 Tensors): 2 Tensors to be added.

        returns:
        value (ndarray): Result of "+" operation on
            input args.
        """
        # Extend this nodes' parents to include new inputs.
        self.parents = list(args)
        # Add the 2 input Tensor's values
        value = self.parents[0].value + self.parents[1].value
        return value

    def backward(self, gradient):
        """
        Accumulates the gradients for "Add" operation from its'
        children/outputs and passes them on to self.parents.

        param:
        gradient (ndarray or scalar): gradient w.r.t output of
        "Add"

        returns:
        None
        """
        # Accumulate gradient
        self.parents[0].grad += gradient
        self.parents[1].grad += gradient


class Sub(Function):
    """
    Subtract parent inputs and return the result.
    """
    def forward(self, *args):
        """
        Forward computation of the binary Subtract operation.

        param:
        args (n=2 Tensors): 2 Tensors to be subtracted.

        returns:
        value (ndarray): Result of "-" operation on
            input args.
        """
        # Extend this nodes' parents to include new inputs.
        self.parents = list(args)
        # Subtract the 2 input Tensor's values
        value = self.parents[0].value - self.parents[1].value
        return value

    def backward(self, gradient):
        """
        Accumulates the gradients for "Subtract" operation from its'
        children/outputs and passes them on to self.parents.

        param:
        gradient (ndarray or scalar): gradient w.r.t output of
        "Sub"

        returns:
        None
        """
        # Accumulate gradient
        self.parents[0].grad += gradient
        self.parents[1].grad -= gradient

class Mul(Function):
    """
    Multiply parent inputs and return the result.
    """
    def forward(self, *args):
        """
        Forward computation of the binary Multiply operation.

        param:
        args (n=2 Tensors): 2 Tensors to be multiplied.

        returns:
        value (ndarray): Result of "*" operation on
            input args.
        """
        # Extend this nodes' parents to include new inputs.
        self.parents = list(args)
        # Multiply the 2 input Tensor's values
        value = self.parents[0].value * self.parents[1].value
        return value

    def backward(self, gradient):
        """
        Accumulates the gradients for "Multiply" operation from its'
        children/outputs and passes them on to self.parents.

        param:
        gradient (ndarray or scalar): gradient w.r.t output of
        "Mul"

        returns:
        None
        """
        # Accumulate gradient
        self.parents[0].grad += np.multiply(gradient, self.parents[1].value)
        self.parents[1].grad += np.multiply(gradient, self.parents[0].value)

class Div(Function):
    """
    Divide parent inputs and return the result.
    """
    def forward(self, *args):
        """
        Forward computation of the binary Divide operation.

        param:
        args (n=2 Tensors): 2 Tensors to be multiplied.

        returns:
        value (ndarray): Result of "/" operation on
            input args.
        """
        # Extend this nodes' parents to include new inputs.
        self.parents = list(args)
        # Divide the 2 input Tensor's values
        value = self.parents[0].value / self.parents[1].value
        return value

    def backward(self, gradient):
        """
        Accumulates the gradients for "Divide" operation from its'
        children/outputs and passes them on to self.parents.

        param:
        gradient (ndarray or scalar): gradient w.r.t output of
        "Div"

        returns:
        None

        """
        # Accumulate gradient
        self.parents[0].grad += np.multiply(
                gradient, self.parents[1].value/self.parents[1].value**2
            )
        self.parents[1].grad -= np.multiply(
                gradient, self.parents[0].value/self.parents[1].value**2
            )

class Sum(Function):
    """
    Implements Sum of a 1xN input vector and return the result of size 1X1.
    """
    def forward(self, *args):
        """
        Forward computation of the unary Sum operation.

        param:
        args (n=1 Tensor): Tensor whose elements are to be added.

        returns:
        value (ndarray): Result of "Sum" operation on
            input args.
        """
        # Extend this nodes' parents to include new inputs.
        self.parents = list(args)
        # Sum input Tensor's values
        value = np.sum(self.parents[0].value)
        return value

    def backward(self, gradient = 1):
        """
        Accumulates the gradients for "Sum" operation from its'
        child/output and passes them on to self.parent.

        param:
        gradient (ndarray or scalar): gradient w.r.t output of
        "Sum"

        returns:
        None
        """
        # Accumulate gradient
        self.parents[0].grad += np.ones(self.parents[0].value.shape)*gradient


class ReLU(Function):
    """
    ReLU parent input and return the result.
    """
    def forward(self, *args):
        """
        Forward computation of the unary ReLU operation.

        param:
        args (n=1 Tensor): Tensor on which ReLU operation
        is to be applied.

        returns:
        value (ndarray): Result of "ReLU" operation on
            input args.
        """
        self.parents = list(args)
        # Extend this nodes' parents to include new inputs.
        # RelU the input Tensors's values
        value = np.maximum(0, self.parents[0].value)
        return value

    def backward(self, gradient):
        """
        Accumulates the gradients for "ReLU" operation from its'
        child/output and passes them on to self.parent.

        param:
        gradient (ndarray or scalar): gradient w.r.t output of
        "ReLU"

        returns:
        None
        """
        # Accumulate gradient
        self.parents[0].grad[
            self.parents[0].value > 0
        ] += gradient[
            self.parents[0].value > 0
        ]

class leakyReLU(Function):
    """
    ReLU parent input and return the result.
    """
    def forward(self, *args):
        """
        Forward computation of the unary ReLU operation.

        param:
        args (n=1 Tensor): Tensor on which ReLU operation
        is to be applied.

        returns:
        value (ndarray): Result of "ReLU" operation on
            input args.
        """
        self.parents = list(args)
        # Extend this nodes' parents to include new inputs.
        # RelU the input Tensors's values
        value = self.parents[0].value
        value[value <= 0] = value[value <= 0] * 0.1
        return value

    def backward(self, gradient):
        """
        Accumulates the gradients for "ReLU" operation from its'
        child/output and passes them on to self.parent.

        param:
        gradient (ndarray or scalar): gradient w.r.t output of
        "ReLU"

        returns:
        None
        """
        # Accumulate gradient
        self.parents[0].grad[self.parents[0].value > 0] += gradient[self.parents[0].value > 0]
        self.parents[0].grad[self.parents[0].value <= 0] += gradient[self.parents[0].value <= 0] * 0.1
        
class Mean(Function):
    """
    Implements Mean of a 1xN input vector and return the result of size 1X1.
    """
    def forward(self, *args):
        """
        Forward computation of the unary Mean operation.

        param:
        args (n=1 Tensor): Tensor on which Mean operation
        is to be applied.

        returns:
        value (ndarray): Result of "Mean" operation on
            input args.
        """
        self.parents = list(args)
        # Extend this nodes' parents to include new inputs.
        # Average the input Tensor's values
        value = np.mean(self.parents[0].value)
        return value

    def backward(self, gradient):
        """
        Accumulates the gradients for "Mean" operation from its'
        child/output and passes them on to self.parent.

        param:
        gradient (ndarray or scalar): gradient w.r.t output of
        "Mean"

        returns:
        None
        """

        # Accumulate gradient
        self.parents[0].grad = (np.ones_like(self.parents[0].value)*gradient)/\
                                    self.parents[0].value.shape[0]

class Pow(Function):
    """
    Implements Power operation on a 1xN input vector.
    Raises the input to an exponent
    """
    def forward(self, exp, *args):
        """
        Forward computation of the bianry Power operation.

        param:
        args (n=1 Tensor): Tensor on which Power operation
        is to be applied.

        exp (int): Exponent of the power function
        returns:

        value (ndarray): Result of "Power" operation on
            input args.
        """
        self.parents = list(args)
        self.exp = exp
        # Extend this nodes' parents to include new inputs.
        # Average the input Tensor's values
        value = np.power(self.parents[0].value, exp)
        return value

    def backward(self, gradient):
        """
        Accumulates the gradients for "Power" operation from its'
        child/output and passes them on to self.parent.

        param:
        gradient (ndarray or scalar): gradient w.r.t output of
        "Pow"

        returns:
        None
        """

        # Accumulate gradient
        self.parents[0].grad += gradient * self.exp *\
                                np.power(self.parents[0].value, self.exp - 1)


class Dot(Function):
    """
    Computes the dot product of the 2 parent inputs.
    """

    def forward(self, *args):
        """
        Forward computation of the bianry Dot operation.

        param:
        args (n=2 Tensors): Tensors on which Dot operation
        is to be performed.

        returns:
        value (ndarray): Result of "Dot" operation on
            input args.
        """
        # Extend this nodes' parents to include new inputs.
        self.parents = list(args)
        # Average the input Tensor's value
        value = np.dot(self.parents[0].value, self.parents[1].value)
        return value

    def backward(self, gradient):
        """
        Accumulates the gradients for "Dot" operation from its'
        children/outputs and passes them on to self.parent.

        param:
        gradient (ndarray or scalar): gradient w.r.t output of
        "Dot"

        returns:
        None
        """
        # Accumulate gradient
        self.parents[0].grad += np.dot(gradient, self.parents[1].value.T)
        self.parents[1].grad += np.dot(self.parents[0].value.T, gradient)


class Sigmoid(Function):
    """
    Implements Sigmoid operation on a 1xN input vector.
    Raises the input to an exponent
    """
    def forward(self, *args):
        """
        Forward computation of the unary Sigmoid operation.

        param:
        args (n=1 Tensor): Tensor on which Sigmoid operation
        is to be applied.

        value (ndarray): Result of "Sigmoid" operation on
            input args.
        """
        self.parents = list(args)

        # Extend this nodes' parents to include new inputs.
        value = 1.0/(1.0 + np.exp(-self.parents[0].value))
        self.sigmoid = value
        return value

    def backward(self, gradient):
        """
        Accumulates the gradients for "Sigmoid" operation from its'
        child/output and passes them on to self.parent.

        param:
        gradient (ndarray or scalar): gradient w.r.t output of
        "Sigmoid"

        returns:
        None
        """

        # Accumulate gradient
        self.parents[0].grad += np.multiply(gradient, self.sigmoid * (1 - self.sigmoid))


class Absolute(Function):
    """
    Implements Absolute operation on a 1xN input vector.
    Raises the input to an exponent
    """
    def forward(self, *args):
        """
        Forward computation of the unary Absolute operation.

        param:
        args (n=1 Tensor): Tensor on which Absolute operation
        is to be applied.

        exp (int): Exponent of the Absolute function
        returns:

        value (ndarray): Result of "Absolute" operation on
            input args.
        """
        self.parents = list(args)
        # Extend this nodes' parents to include new inputs.
        # Average the input Tensor's values
        value = np.abs(self.parents[0].value)
        return value

    def backward(self, gradient):
        """
        Accumulates the gradients for "Power" operation from its'
        child/output and passes them on to self.parent.

        param:
        gradient (ndarray or scalar): gradient w.r.t output of
        "Pow"

        returns:
        None
        """
        value = np.ones(gradient.shape)
        value[self.parents[0].value < 0] = -1
        # Accumulate gradient
        self.parents[0].grad += gradient*value





