from network import *
import unittest
import torch

class ConvTests(unittest.TestCase):

    def setUp(self):
        self._started_at = time.time()

    def tearDown(self):
        elapsed = time.time() - self._started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 2)))

    def test_forward_1(self):
        a = np.random.rand(50, 1, 50, 50)
        b = np.random.rand(20, 1, 2, 2)
        # Implementation
        result_1 = Tensor(a).conv2d(Tensor(b)).value
        # Torch
        result_2 = torch.nn.functional.conv2d(torch.tensor(a), torch.tensor(b), stride=1).numpy()
        self.assertEqual(np.allclose(result_1, result_2), True)

    def test_forward_2(self):
        a = np.random.rand(50, 1, 28, 28)
        b = np.random.rand(50, 1, 3, 4)
        # Implementation
        result_1 = Tensor(a).conv2d(Tensor(b)).value
        # Torch
        result_2 = torch.nn.functional.conv2d(torch.tensor(a), torch.tensor(b), stride=1).numpy()

        self.assertEqual(np.allclose(result_1, result_2), True)

    def test_forward_3(self):
        a = np.random.rand(20, 3, 28, 30)
        b = np.random.rand(50, 3, 4, 4)
        # Implementation
        result_1 = Tensor(a).conv2d(Tensor(b)).value
        # Torch
        result_2 = torch.nn.functional.conv2d(torch.tensor(a), torch.tensor(b), stride=1).numpy()

        self.assertEqual(np.allclose(result_1, result_2), True)

    def test_backward_1(self):
        a = np.random.rand(50, 2, 30, 30)
        b = np.random.rand(20, 2, 2, 2)
        # Implementation
        inp_1 = Tensor(a)
        kernel_1 = Tensor(b)
        res_1 = inp_1.conv2d(kernel_1)
        mse_1 = res_1.sum()
        mse_1.backward()
        # Torch
        inp_2 = torch.tensor(a, requires_grad=True)
        kernel_2 = torch.tensor(b, requires_grad=True)
        res_2 = torch.nn.functional.conv2d(inp_2, kernel_2, stride=1)
        mse_2 = torch.sum(res_2)
        mse_2.backward()

        self.assertEqual(np.allclose(inp_1.grad, inp_2.grad.numpy()), True)
        self.assertEqual(np.allclose(kernel_1.grad, kernel_2.grad.numpy()), True)

    def test_backward_2(self):
        a = np.random.rand(3, 1, 3, 3)
        b = np.random.rand(2, 1, 2, 2)
        # Implementation
        inp_1 = Tensor(a)
        kernel_1 = Tensor(b)
        res_1 = inp_1.conv2d(kernel_1)
        y_1 = Tensor(np.ones_like(res_1.value))
        mse_1 = (y_1 - res_1).pow(2).sum()
        mse_1.backward()
        # Torch
        inp_2 = torch.tensor(a, requires_grad=True)
        kernel_2 = torch.tensor(b, requires_grad=True)
        res_2 = torch.nn.functional.conv2d(inp_2, kernel_2, stride=1)
        y_2 = torch.tensor(np.ones_like(res_2.data))
        mse_2 = torch.sum((y_2 - res_2) ** 2)
        mse_2.backward()

        self.assertEqual(np.allclose(inp_1.grad, inp_2.grad.numpy()), True)
        self.assertEqual(np.allclose(kernel_1.grad, kernel_2.grad.numpy()), True)

    def test_backward_3(self):
        a = np.random.rand(4, 3, 20, 20)
        b = np.random.rand(10, 3, 3, 4)
        # Implementation
        inp_1 = Tensor(a)
        kernel_1 = Tensor(b)
        res_1 = inp_1.conv2d(kernel_1)
        y_1 = Tensor(np.ones_like(res_1.value))
        mse_1 = (y_1 - res_1).pow(2).sum()
        mse_1.backward()
        # Torch
        inp_2 = torch.tensor(a, requires_grad=True)
        kernel_2 = torch.tensor(b, requires_grad=True)
        res_2 = torch.nn.functional.conv2d(inp_2, kernel_2, stride=1)
        y_2 = torch.tensor(np.ones_like(res_2.data))
        mse_2 = torch.sum((y_2 - res_2) ** 2)
        mse_2.backward()

        self.assertEqual(np.allclose(inp_1.grad, inp_2.grad.numpy()), True)
        self.assertEqual(np.allclose(kernel_1.grad, kernel_2.grad.numpy()), True)

    def test_layers_4(self):
        a = np.random.rand(2, 2, 8, 8)
        b = np.random.rand(3, 2, 4, 4)
        c = np.random.rand(4, 3, 2, 2)
        # Implementation
        inp_1 = Tensor(a)
        kernel_1 = Tensor(b)
        kernel_11 = Tensor(c)
        res_1 = inp_1.conv2d(kernel_1)
        res_11 = res_1.conv2d(kernel_11)
        y_1 = Tensor(np.ones_like(res_11.value))
        mse = (y_1 - res_11).pow(2).sum()
        mse.backward()
        # Torch
        inp_2 = torch.tensor(a, requires_grad=True)
        kernel_2 = torch.tensor(b, requires_grad=True)
        kernel_22 = torch.tensor(c, requires_grad=True)
        res_2 = torch.nn.functional.conv2d(inp_2, kernel_2, stride=1)
        res_22 = torch.nn.functional.conv2d(res_2, kernel_22, stride=1)
        y_2 = torch.tensor(np.ones_like(res_22.data))
        mse = torch.sum((y_2 - res_22) ** 2)
        mse.backward()

        self.assertEqual(np.allclose(inp_1.grad, inp_2.grad.numpy()), True)
        self.assertEqual(np.allclose(kernel_1.grad, kernel_2.grad.numpy()), True)
        self.assertEqual(np.allclose(kernel_11.grad, kernel_22.grad.numpy()), True)

    def test_time(self):
        a = np.random.rand(100, 1, 28, 28)
        b = np.random.rand(20, 1, 14, 14)
        c = np.random.rand(30, 20, 7, 7)
        # Implementation
        inp_1 = Tensor(a)
        kernel_1 = Tensor(b)
        kernel_11 = Tensor(c)
        res_1 = inp_1.conv2d(kernel_1)
        res_11 = res_1.conv2d(kernel_11)
        y_1 = Tensor(np.ones_like(res_11.value))
        mse_1 = (y_1 - res_11).pow(2).sum()
        mse_1.backward()
        # Torch
        inp_2 = torch.tensor(a, requires_grad=True)
        kernel_2 = torch.tensor(b, requires_grad=True)
        kernel_22 = torch.tensor(c, requires_grad=True)
        res_2 = torch.nn.functional.conv2d(inp_2, kernel_2, stride=1)
        res_22 = torch.nn.functional.conv2d(res_2, kernel_22, stride=1)
        y_2 = torch.tensor(np.ones_like(res_22.data))
        mse_2 = torch.sum((y_2 - res_22) ** 2)
        mse_2.backward()

        self.assertEqual(np.allclose(inp_1.grad, inp_2.grad.numpy()), True)
        self.assertEqual(np.allclose(kernel_1.grad, kernel_2.grad.numpy()), True)
        self.assertEqual(np.allclose(kernel_11.grad, kernel_22.grad.numpy()), True)

    def test_reshape(self):
        a = np.random.rand(2, 4, 3, 3)
        b = np.random.rand(72, 1)
        # Implementation
        inp_1 = Tensor(a)
        inp_11 = Tensor(b)
        reshape_1 = inp_1.reshape(b.shape)
        loss_1 = (inp_11 + reshape_1).sum()
        loss_1.backward()
        # Torch
        inp_2 = torch.tensor(a, requires_grad=True)
        inp_22 = torch.tensor(b, requires_grad=True)
        reshape_2 = torch.sum(inp_2.view(b.shape) + inp_22)
        reshape_2.backward()

        self.assertEqual(np.allclose(inp_1.grad, inp_2.grad.numpy()), True)
        self.assertEqual(np.allclose(inp_11.grad, inp_22.grad.numpy()), True)

class NetworkTests(unittest.TestCase):
    def test_mnist_dataset_initialization(self):
        with self.assertRaises(ValueError):
            dataset = Dataset('mist')
    def test_mnist_network_initialization(self):
        with self.assertRaises(ValueError):
            network = ConvNetwork('mist')


if __name__ == '__main__':
    unittest.main(exit=False)
