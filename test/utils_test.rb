require_relative "test_helper"

class UtilsTest < Minitest::Test
  def test_save_image
    trainset = TorchVision::Datasets::MNIST.new(root, train: true, download: true, transform: TorchVision::Transforms::ToTensor.new)
    trainloader = Torch::Utils::Data::DataLoader.new(trainset, batch_size: 64)
    images, _labels = trainloader.first
    TorchVision::Utils.save_image(images[0...25], "#{Dir.tmpdir}/image.png", nrow: 5, normalize: true)
  end
end
