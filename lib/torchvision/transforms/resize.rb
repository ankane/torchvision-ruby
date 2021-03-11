module TorchVision
  module Transforms
    class Resize < Torch::NN::Module
      def initialize(size)
        @size = size
      end

      def forward(img)
        F.resize(img, @size)
      end
    end
  end
end
