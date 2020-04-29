require_relative "test_helper"

class ModelsTest < Minitest::Test
  def test_resnet
    net = TorchVision::Models::ResNet18.new
  end
end
