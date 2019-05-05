from functions import Function
from numba import jit
import np_utils
import numpy as np

class Convolution2D(Function):

    @staticmethod
    @jit(nopython=True)
    def forward_helper_func(i,k):
        """
        Example of an accelerated function, Notice the Numba jit decorator on top.
        """
        # calculate metadata associated with the input and the kernel

        num_batches = i.shape[0]
        num_channels = i.shape[1]
        num_input_rows = i.shape[2]
        num_input_columns = i.shape[3]
        num_filters = k.shape[0]
        num_kernel_rows = k.shape[2]
        num_kernel_columns = k.shape[3]
        num_output_rows = num_input_rows - num_kernel_rows + 1
        num_output_columns = num_input_columns - num_kernel_columns + 1

        # initialize a numpy array to hold the output with the correct size
        value = np.zeros((num_batches,num_filters,num_output_rows, num_output_columns))

        # for every entry in the output:
            # that entry is calculated by doing a piecewise multiplication between
            # the kernel and the input window which is the same size as the kernel.

        for batch in range(0,num_batches):
            for filter in range(0,num_filters):
                for row in range(0,num_output_rows):
                    for col in range(0,num_output_columns):
                        value[batch,filter,row,col] = np.sum(np.multiply(k[filter,:,:,:], i[batch,:,row:row+num_kernel_rows,col:col+num_kernel_columns]))
        return value

    def forward(self, stride, padding, *args):
        """
        Forward pass of the convolution operation between two four dimensional tensors.
        :param stride: Convolution stride, defaults to 1.
        :param padding: Convolution padding, defaults to 0.
        :param args: Operands of convolution operation (input(batch_size, in_channels, H, W), kernel(out_channels, in_channels, Hk, Wk)).
        :return: Output of the convolution operation.
        """
        # All we do here is save the parents and call the accelerated helper function
        # passing the value of the input and the kernel.
        self.parents = list(args)
        value = Convolution2D.forward_helper_func(self.parents[0].value,self.parents[1].value)
        return value

    @staticmethod
    @jit(nopython=True)
    def backward_helper_func(i,k,gradient):
        """
        Example of an accelerated function, Notice the Numba jit decorator on top.
        """
        # Calculate metadata for the input, kernel
        num_batches = i.shape[0]
        num_channels = i.shape[1]
        num_input_rows = i.shape[2]
        num_input_columns = i.shape[3]
        num_filters = k.shape[0]
        num_kernel_rows = k.shape[2]
        num_kernel_columns = k.shape[3]
        num_output_rows = num_input_rows - num_kernel_rows + 1
        num_output_columns = num_input_columns - num_kernel_columns + 1

        # the gradients for the input and the kernel will be the same sizes as the
        # original tensors
        igrad = np.zeros(i.shape)
        kgrad = np.zeros(k.shape)

        # for each entry in the kernel:
            # the gradient is the upstream gradient times the input window

        for channel in range(0,num_channels):
            for row in range(0,num_kernel_rows):
                for col in range(0,num_kernel_columns):
                    for filter in range(0,num_filters):
                        kgrad[filter,channel,row,col] = np.sum(np.multiply(gradient[:,filter,:,:], i[:,channel,row:num_input_rows-(num_kernel_rows - (row + 1)) ,col:num_input_columns-(num_kernel_columns - (col + 1))]))

        # for each entry in the input:
            # the gradient is the upstream gradient times the kernel.
            # here we accumulate the gradients rather than overwritting them because
            # each entry in the input will have multiple gradients to accumulate from
            # different kernel entries

        for batch in range(0,num_batches):
            for filter in range(0,num_filters):
                for row in range(0,num_output_rows):
                    for col in range(0,num_output_columns):
                        igrad[batch,:,row:row + num_kernel_rows,col:col + num_kernel_columns] += np.multiply(gradient[batch,filter,row,col], k[filter,:,:,:])

        return igrad,kgrad

    def backward(self, gradient):
        """
        Sets the gradients for operands of convolution operation.
        :param gradient: Upstream gradient.
        """
        # All we do here is call the accelerated helper function and update the gradients
        igrad,kgrad = Convolution2D.backward_helper_func(self.parents[0].value,self.parents[1].value,gradient)
        self.parents[0].grad += igrad
        self.parents[1].grad += kgrad

class Reshape(Function):
    def forward(self, shape, *args):
        """
        Forward pass of the reshape operation on a tensor
        :param shape: tuple of required dimension.
        :param args: Input tensor to be reshaped.
        :return: reshaped tensor.
        """
        # Don't reshape the original input, but reshape a copy and return it.
        self.parents = list(args)
        reshaped_value = np.reshape(self.parents[0].value,shape)
        return reshaped_value

    def backward(self, gradient):
        """
        Sets the gradient for input of reshape operation.
        :param gradient: Upstream gradient.
        """
        # Because the reshape function doesn't do any computation the backward pass
        # just passes back the gradient from upstream in the parent shape.
        self.parents[0].grad += np.reshape(gradient,self.parents[0].value.shape)
