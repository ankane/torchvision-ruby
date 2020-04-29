require_relative "test_helper"

class ModelsTest < Minitest::Test
  def test_resnet
    transform = TorchVision::Transforms::Compose.new([
      TorchVision::Transforms::ToTensor.new,
      TorchVision::Transforms::Normalize.new([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    trainset = TorchVision::Datasets::CIFAR10.new(root, train: true, download: true, transform: transform)
    trainloader = Torch::Utils::Data::DataLoader.new(trainset, batch_size: 4)

    net = TorchVision::Models::ResNet18.new
    trainloader.each do |data, target|
      net.call(data)
      break
    end
  end
end
