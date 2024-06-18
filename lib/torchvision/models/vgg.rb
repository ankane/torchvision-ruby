module TorchVision
  module Models
    class VGG < Torch::NN::Module
      MODEL_URLS = {
        "vgg11" => "https://download.pytorch.org/models/vgg11-8a719046.pth",
        "vgg13" => "https://download.pytorch.org/models/vgg13-19584684.pth",
        "vgg16" => "https://download.pytorch.org/models/vgg16-397923af.pth",
        "vgg19" => "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
        "vgg11_bn" => "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
        "vgg13_bn" => "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
        "vgg16_bn" => "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
        "vgg19_bn" => "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth"
      }

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

      CFGS = {
        "A" => [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "B" => [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "D" => [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
        "E" => [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
      }

      def self.make_model(arch, cfg, batch_norm, pretrained: false, **kwargs)
        kwargs[:init_weights] = false if pretrained
        model = VGG.new(make_layers(CFGS[cfg], batch_norm), **kwargs)
        if pretrained
          url = MODEL_URLS[arch]
          state_dict = Torch::Hub.load_state_dict_from_url(url)
          model.load_state_dict(state_dict)
        end
        model
      end

      def self.make_layers(cfg, batch_norm)
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
