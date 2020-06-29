module TorchVision
  module Models
    class AlexNet < Torch::NN::Module
      def initialize(num_classes: 1000)
        super()
        @features = Torch::NN::Sequential.new(
          Torch::NN::Conv2d.new(3, 64, 11, stride: 4, padding: 2),
          Torch::NN::ReLU.new(inplace: true),
          Torch::NN::MaxPool2d.new(3, stride: 2),
          Torch::NN::Conv2d.new(64, 192, 5, padding: 2),
          Torch::NN::ReLU.new(inplace: true),
          Torch::NN::MaxPool2d.new(3, stride: 2),
          Torch::NN::Conv2d.new(192, 384, 3, padding: 1),
          Torch::NN::ReLU.new(inplace: true),
          Torch::NN::Conv2d.new(384, 256, 3, padding: 1),
          Torch::NN::ReLU.new(inplace: true),
          Torch::NN::Conv2d.new(256, 256, 3, padding: 1),
          Torch::NN::ReLU.new(inplace: true),
          Torch::NN::MaxPool2d.new(3, stride: 2),
        )
        @avgpool = Torch::NN::AdaptiveAvgPool2d.new([6, 6])
        @classifier = Torch::NN::Sequential.new(
          Torch::NN::Dropout.new,
          Torch::NN::Linear.new(256 * 6 * 6, 4096),
          Torch::NN::ReLU.new(inplace: true),
          Torch::NN::Dropout.new,
          Torch::NN::Linear.new(4096, 4096),
          Torch::NN::ReLU.new(inplace: true),
          Torch::NN::Linear.new(4096, num_classes)
        )
      end

      def forward(x)
        x = @features.call(x)
        x = @avgpool.call(x)
        x = Torch.flatten(x, 1)
        x = @classifier.call(x)
        x
      end
    end
  end
end
