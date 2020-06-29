require_relative "test_helper"

class ModelsTest < Minitest::Test
  def test_alexnet
    net = TorchVision::Models::AlexNet.new
    assert_equal 24, net.modules.size
    assert_equal 16, net.parameters.size
  end

  def test_resnet18
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

  def test_resnet34
    net = TorchVision::Models::ResNet34.new
    assert_equal 116, net.modules.size
    assert_equal 110, net.parameters.size
  end

  def test_resnet50
    net = TorchVision::Models::ResNet50.new
    assert_equal 151, net.modules.size
    assert_equal 161, net.parameters.size
  end

  def test_resnet101
    net = TorchVision::Models::ResNet101.new
    assert_equal 287, net.modules.size
    assert_equal 314, net.parameters.size
  end

  def test_resnet152
    net = TorchVision::Models::ResNet152.new
    assert_equal 423, net.modules.size
    assert_equal 467, net.parameters.size
  end

  def test_resnext50_32x4d
    net = TorchVision::Models::ResNext50_32x4d.new
    assert_equal 151, net.modules.size
    assert_equal 161, net.parameters.size
  end

  def test_resnext101_32x8d
    net = TorchVision::Models::ResNext101_32x8d.new
    assert_equal 287, net.modules.size
    assert_equal 314, net.parameters.size
  end

  def test_vgg11
    net = TorchVision::Models::VGG11.new
    assert_equal 32, net.modules.size
    assert_equal 22, net.parameters.size
  end

  def test_vgg11_bn
    net = TorchVision::Models::VGG11BN.new
    assert_equal 40, net.modules.size
    assert_equal 38, net.parameters.size
  end

  def test_vgg13
    net = TorchVision::Models::VGG13.new
    assert_equal 36, net.modules.size
    assert_equal 26, net.parameters.size
  end

  def test_vgg13_bn
    net = TorchVision::Models::VGG13BN.new
    assert_equal 46, net.modules.size
    assert_equal 46, net.parameters.size
  end

  def test_vgg16
    net = TorchVision::Models::VGG16.new
    assert_equal 42, net.modules.size
    assert_equal 32, net.parameters.size
  end

  def test_vgg16_bn
    net = TorchVision::Models::VGG16BN.new
    assert_equal 55, net.modules.size
    assert_equal 58, net.parameters.size
  end

  def test_vgg19
    net = TorchVision::Models::VGG19.new
    assert_equal 48, net.modules.size
    assert_equal 38, net.parameters.size
  end

  def test_vgg19_bn
    net = TorchVision::Models::VGG19BN.new
    assert_equal 64, net.modules.size
    assert_equal 70, net.parameters.size
  end

  def test_wide_resnet50_2
    net = TorchVision::Models::WideResNet50_2.new
    assert_equal 151, net.modules.size
    assert_equal 161, net.parameters.size
  end

  def test_wide_resnet101_2
    net = TorchVision::Models::WideResNet101_2.new
    assert_equal 287, net.modules.size
    assert_equal 314, net.parameters.size
  end
end
