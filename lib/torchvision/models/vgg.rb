module TorchVision
  module Models
    class VGG < Torch::NN::Module
      def initialize(features, num_classes: 1000, init_weights: true)
        super()
        @features = features
        @avgpool = Torch::NN::AdaptiveAvgPool2d.new([7, 7])
        @classifier = Torch::NN::Sequential.new(
          Torch::NN::Linear.new(512 * 7 * 7, 4096),
          Torch::NN::ReLU.new(inplace: true),
          Torch::NN::Dropout.new,
          Torch::NN::Linear.new(4096, 4096),
          Torch::NN::ReLU.new(inplace: true),
          Torch::NN::Dropout.new,
          Torch::NN::Linear.new(4096, num_classes)
        )
        _initialize_weights if init_weights
      end

      def forward(x)
        x = @features.call(x)
        x = @avgpool.call(x)
        x = Torch.flatten(x, 1)
        x = @classifier.call(x)
        x
      end

      def _initialize_weights
        modules.each do |m|
          case m
          when Torch::NN::Conv2d
            Torch::NN::Init.kaiming_normal!(m.weight, mode: "fan_out", nonlinearity: "relu")
            Torch::NN::Init.constant!(m.bias, 0) if m.bias
          when Torch::NN::BatchNorm2d
            Torch::NN::Init.constant!(m.weight, 1)
            Torch::NN::Init.constant!(m.bias, 0)
          when Torch::NN::Linear
            Torch::NN::Init.normal!(m.weight, mean: 0, std: 0.01)
            Torch::NN::Init.constant!(m.bias, 0)
          end
        end
      end

      def self.make_layers(cfg, batch_norm)
        cfgs = {
          "A" => [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
          "B" => [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
          "D" => [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
          "E" => [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
        }
        cfg = cfgs[cfg]

        layers = []
        in_channels = 3
        cfg.each do |v|
          if v == "M"
            layers += [Torch::NN::MaxPool2d.new(2, stride: 2)]
          else
            conv2d = Torch::NN::Conv2d.new(in_channels, v, 3, padding: 1)
            if batch_norm
              layers += [conv2d, Torch::NN::BatchNorm2d.new(v), Torch::NN::ReLU.new(inplace: true)]
            else
              layers += [conv2d, Torch::NN::ReLU.new(inplace: true)]
            end
            in_channels = v
          end
        end
        Torch::NN::Sequential.new(*layers)
      end
    end
  end
end
