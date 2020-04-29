module TorchVision
  module Models
    class BasicBlock < Torch::NN::Module
      def initialize(inplanes, planes, stride: 1, downsample: nil, groups: 1, base_width: 64, dilation: 1, norm_layer: nil)
        super()
        norm_layer ||= Torch::NN::BatchNorm2d
        if groups != 1 || base_width != 64
          raise ArgumentError, "BasicBlock only supports groups=1 and base_width=64"
        end
        if dilation > 1
          raise NotImplementedError, "Dilation > 1 not supported in BasicBlock"
        end
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        @conv1 = Torch::NN::Conv2d.new(inplanes, planes, 3, stride: stride, padding: 1, groups: 1, bias: false, dilation: 1)
        @bn1 = norm_layer.new(planes)
        @relu = Torch::NN::ReLU.new(inplace: true)
        @conv1 = Torch::NN::Conv2d.new(planes, planes, 3, stride: 1, padding: 1, groups: 1, bias: false, dilation: 1)
        @bn2 = norm_layer.new(planes)
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

        identity = @downsample.call(x) if @downsample

        out += identity
        out = @relu.call(out)

        out
      end

      def self.expansion
        1
      end
    end
  end
end
