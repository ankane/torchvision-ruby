require_relative "test_helper"

class ModelsTest < Minitest::Test
  def test_alexnet
    net = TorchVision::Models::AlexNet.new
    assert_equal 24, net.modules.size
  end

  def test_resnet
    Torch.manual_seed(1)

    transform = TorchVision::Transforms::Compose.new([
      TorchVision::Transforms::ToTensor.new,
      TorchVision::Transforms::Normalize.new([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    trainset = TorchVision::Datasets::CIFAR10.new(root, train: true, download: true, transform: transform)
    trainloader = Torch::Utils::Data::DataLoader.new(trainset, batch_size: 4)

    net = TorchVision::Models::ResNet18.new
    assert_equal 68, net.modules.size
    assert net.named_modules.keys.include?("layer2.0.downsample.0")
    trainloader.each do |data, target|
      net.call(data)
      break
    end
  end
end
