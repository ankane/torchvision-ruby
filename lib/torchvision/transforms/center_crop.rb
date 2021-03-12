module TorchVision
  module Transforms
    class CenterCrop < Torch::NN::Module
      def initialize(size)
        @size = size
      end

      def forward(img)
        F.center_crop(img, @size)
      end
    end
  end
end
