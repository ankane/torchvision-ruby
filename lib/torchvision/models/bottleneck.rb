module TorchVision
  module Models
    class Bottleneck < Torch::NN::Module
      def initialize(inplanes, planes, stride: 1, downsample: nil, groups: 1, base_width: 64, dilation: 1, norm_layer: nil)
        super()
        norm_layer ||= Torch::NN::BatchNorm2d
        width = (planes * (base_width / 64.0)).to_i * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        @conv1 = Torch::NN::Conv2d.new(inplanes, width, 1, stride: 1, bias: false)
        @bn1 = norm_layer.new(width)
        @conv2 = Torch::NN::Conv2d.new(width, width, 3, stride: stride, padding: dilation, groups: groups, bias: false, dilation: dilation)
        @bn2 = norm_layer.new(width)
        @conv3 = Torch::NN::Conv2d.new(width, planes * self.class.expansion, 1, stride: 1, bias: false)
        @bn3 = norm_layer.new(planes * self.class.expansion)
        @relu = Torch::NN::ReLU.new(inplace: true)
        @downsample = downsample
        @stride = stride
      end

      def forward(x)
        identity = x

        out = @conv1.call(x)
        out = @bn1.call(out)
        out = @relu.call(out)

        out = @conv2.call(out)
        out = @bn2.call(out)
        out = @relu.call(out)

        out = @conv3.call(out)
        out = @bn3.call(out)

        identity = @downsample.call(x) if @downsample

        out += identity
        out = @relu.call(out)

        out
      end

      def self.expansion
        4
      end
    end
  end
end
