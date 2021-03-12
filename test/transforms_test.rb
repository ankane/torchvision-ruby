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

  def test_random_horizontal_flip
    transform = TorchVision::Transforms::RandomHorizontalFlip.new
    x = Torch.tensor([[1, 2, 3], [4, 5, 6]])
    result = transform.call(x).to_a
    if result[0][0] == 1
      assert_equal [[1, 2, 3], [4, 5, 6]], result
    else
      assert_equal [[3, 2, 1], [6, 5, 4]], result
    end
  end

  def test_random_vertical_flip
    transform = TorchVision::Transforms::RandomVerticalFlip.new
    x = Torch.tensor([[1, 2, 3], [4, 5, 6]])
    result = transform.call(x).to_a
    if result[0][0] == 1
      assert_equal [[1, 2, 3], [4, 5, 6]], result
    else
      assert_equal [[4, 5, 6], [1, 2, 3]], result
    end
  end

  def test_hflip_tensor
    x = Torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert_equal [[3, 2, 1], [6, 5, 4]], TorchVision::Transforms::F.hflip(x).to_a
  end

  def test_hflip_image
    # TODO
  end

  def test_vflip_tensor
    x = Torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert_equal [[4, 5, 6], [1, 2, 3]], TorchVision::Transforms::F.vflip(x).to_a
  end

  def test_vflip_image
    # TODO
  end

  def test_crop_tensor
    x = Torch.arange(0, 100).reshape(10, 10)
    assert_equal [[54, 55], [64, 65], [74, 75]], TorchVision::Transforms::F.crop(x, 5, 4, 3, 2).to_a
  end

  def test_crop_image
    # TODO
  end

  def test_center_crop_tensor
    x = Torch.arange(0, 100).reshape(10, 10)
    assert_equal [[44, 45], [54, 55]], TorchVision::Transforms::F.center_crop(x, 2).to_a
  end

  def test_center_crop_image
    # TODO
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
