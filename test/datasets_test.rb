require_relative "test_helper"

class DatasetsTest < Minitest::Test
  def test_mnist
    trainset = TorchVision::Datasets::MNIST.new(root, train: true, download: true)
    assert_equal 60000, trainset.size
    assert_kind_of Torch::Tensor, trainset[0][0]
    assert_equal 5, trainset[0][1]

    testset = TorchVision::Datasets::MNIST.new(root, train: false, download: true)
    assert_equal 10000, testset.size
    assert_kind_of Torch::Tensor, testset[0][0]
    assert_equal 7, testset[0][1]
  end

  def test_cifar10
    trainset = TorchVision::Datasets::CIFAR10.new(root, train: true, download: true)
    assert_equal 50000, trainset.size
    # just test data for now
    data = trainset.instance_variable_get("@data")
    assert_equal [50000, 32, 32, 3], data.shape
    assert_equal [59, 62, 63], data[0..2].to_a
    assert_equal 6, trainset[0][1]

    testset = TorchVision::Datasets::CIFAR10.new(root, train: false, download: true)
    assert_equal 10000, testset.size
  end

  def test_missing
    error = assert_raises(TorchVision::Error) do
      TorchVision::Datasets::MNIST.new(Dir.mktmpdir)
    end
    assert_equal "Dataset not found. You can use download: true to download it", error.message
  end
end
