require_relative "test_helper"

class TransformsTest < Minitest::Test
  def test_mnist
    transform = TorchVision::Transforms::Compose.new([
      TorchVision::Transforms::ToTensor.new,
      TorchVision::Transforms::Normalize.new([0.1307], [0.3081]),
    ])
    trainset = TorchVision::Datasets::MNIST.new(root, train: true, download: true, transform: transform)
    assert_equal [1, 28, 28], trainset[0][0].shape
  end

  def test_cifar10
    transform = TorchVision::Transforms::Compose.new([
      TorchVision::Transforms::ToTensor.new,
      TorchVision::Transforms::Normalize.new([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    trainset = TorchVision::Datasets::CIFAR10.new(root, train: true, download: true, transform: transform)
    assert_equal [3, 32, 32], trainset[0][0].shape
  end
end
