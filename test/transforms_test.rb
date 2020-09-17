require_relative "test_helper"

class TransformsTest < Minitest::Test
  def test_compose
    x = Numo::NArray.cast([[[0.0], [9], [-9]]])
    transform = TorchVision::Transforms::Compose.new([
      TorchVision::Transforms::ToTensor.new,
      TorchVision::Transforms::Normalize.new([0], [3])
    ])
    assert_equal [[[0, 3, -3]]], transform.call(x).to_a
  end

  def test_normalize
    transform = TorchVision::Transforms::Normalize.new([0], [3])
    x = Torch.tensor([[[0.0], [9], [-9]]])
    assert_equal [[[0], [3], [-3]]], transform.call(x).to_a
  end

  def test_to_tensor
    transform = TorchVision::Transforms::ToTensor.new
    x = Numo::NArray.cast([[1, 2, 3], [4, 5, 6]])
    assert_equal [[[1, 2, 3], [4, 5, 6]]], transform.call(x).to_a
  end

  def test_mnist
    transform = TorchVision::Transforms::Compose.new([
      TorchVision::Transforms::ToTensor.new,
      TorchVision::Transforms::Normalize.new([0.1307], [0.3081])
    ])
    trainset = TorchVision::Datasets::MNIST.new(root, train: true, download: true, transform: transform)
    assert_equal [1, 28, 28], trainset[0][0].shape
  end

  def test_cifar10
    transform = TorchVision::Transforms::Compose.new([
      TorchVision::Transforms::ToTensor.new,
      TorchVision::Transforms::Normalize.new([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    trainset = TorchVision::Datasets::CIFAR10.new(root, train: true, download: true, transform: transform)
    assert_equal [3, 32, 32], trainset[0][0].shape
  end
end
