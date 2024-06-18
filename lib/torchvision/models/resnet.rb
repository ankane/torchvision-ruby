module TorchVision
  module Models
    class ResNet < Torch::NN::Module
      MODEL_URLS = {
        "resnet18" => "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "resnet34" => "https://download.pytorch.org/models/resnet34-b627a593.pth",
        "resnet50" => "https://download.pytorch.org/models/resnet50-0676ba61.pth",
        "resnet101" => "https://download.pytorch.org/models/resnet101-63fe2227.pth",
        "resnet152" => "https://download.pytorch.org/models/resnet152-394f9c45.pth",
        "resnext50_32x4d" => "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
        "resnext101_32x8d" => "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
        "wide_resnet50_2" => "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
        "wide_resnet101_2" => "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth"
      }

      def initialize(block, layers, num_classes = 1000, zero_init_residual: false,
        groups: 1, width_per_group: 64, replace_stride_with_dilation: nil, norm_layer: nil)

        super()
        norm_layer ||= Torch::NN::BatchNorm2d
        @norm_layer = norm_layer

        @inplanes = 64
        @dilation = 1
        if replace_stride_with_dilation.nil?
          # each element in the tuple indicates if we should replace
          # the 2x2 stride with a dilated convolution instead
          replace_stride_with_dilation = [false, false, false]
        end
        if replace_stride_with_dilation.length != 3
          raise ArgumentError, "replace_stride_with_dilation should be nil or a 3-element tuple, got #{replace_stride_with_dilation}"
        end
        @groups = groups
        @base_width = width_per_group
        @conv1 = Torch::NN::Conv2d.new(3, @inplanes, 7, stride: 2, padding: 3, bias: false)
        @bn1 = norm_layer.new(@inplanes)
        @relu = Torch::NN::ReLU.new(inplace: true)
        @maxpool = Torch::NN::MaxPool2d.new(3, stride: 2, padding: 1)
        @layer1 = _make_layer(block, 64, layers[0])
        @layer2 = _make_layer(block, 128, layers[1], stride: 2, dilate: replace_stride_with_dilation[0])
        @layer3 = _make_layer(block, 256, layers[2], stride: 2, dilate: replace_stride_with_dilation[1])
        @layer4 = _make_layer(block, 512, layers[3], stride: 2, dilate: replace_stride_with_dilation[2])
        @avgpool = Torch::NN::AdaptiveAvgPool2d.new([1, 1])
        @fc = Torch::NN::Linear.new(512 * block.expansion, num_classes)

        modules.each do |m|
          case m
          when Torch::NN::Conv2d
            Torch::NN::Init.kaiming_normal!(m.weight, mode: "fan_out", nonlinearity: "relu")
          when Torch::NN::BatchNorm2d, Torch::NN::GroupNorm
            Torch::NN::Init.constant!(m.weight, 1)
            Torch::NN::Init.constant!(m.bias, 0)
          end
        end

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual
          modules.each do |m|
            case m
            when Bottleneck
              Torch::NN::Init.constant!(m.bn3.weight, 0)
            when BasicBlock
              Torch::NN::Init.constant!(m.bn2.weight, 0)
            end
          end
        end
      end

      def _make_layer(block, planes, blocks, stride: 1, dilate: false)
        norm_layer = @norm_layer
        downsample = nil
        previous_dilation = @dilation
        if dilate
          @dilation *= stride
          stride = 1
        end
        if stride != 1 || @inplanes != planes * block.expansion
          downsample = Torch::NN::Sequential.new(
            Torch::NN::Conv2d.new(@inplanes, planes * block.expansion, 1, stride: stride, bias: false),
            norm_layer.new(planes * block.expansion)
          )
        end

        layers = []
        layers << block.new(@inplanes, planes, stride: stride, downsample: downsample, groups: @groups, base_width: @base_width, dilation: previous_dilation, norm_layer: norm_layer)
        @inplanes = planes * block.expansion
        (blocks - 1).times do
          layers << block.new(@inplanes, planes, groups: @groups, base_width: @base_width, dilation: @dilation, norm_layer: norm_layer)
        end

        Torch::NN::Sequential.new(*layers)
      end

      def _forward_impl(x)
        x = @conv1.call(x)
        x = @bn1.call(x)
        x = @relu.call(x)
        x = @maxpool.call(x)

        x = @layer1.call(x)
        x = @layer2.call(x)
        x = @layer3.call(x)
        x = @layer4.call(x)

        x = @avgpool.call(x)
        x = Torch.flatten(x, 1)
        x = @fc.call(x)

        x
      end

      def forward(x)
        _forward_impl(x)
      end

      def self.make_model(arch, block, layers, pretrained: false, **kwargs)
        model = ResNet.new(block, layers, **kwargs)
        if pretrained
          url = MODEL_URLS[arch]
          state_dict = Torch::Hub.load_state_dict_from_url(url)
          model.load_state_dict(state_dict)
        end
        model
      end
    end
  end
end
