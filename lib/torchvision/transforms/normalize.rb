module TorchVision
  module Transforms
    class Normalize < Torch::NN::Module
      def initialize(mean, std, inplace: false)
        @mean = mean
        @std = std
        @inplace = inplace
      end

      def forward(tensor)
        F.normalize(tensor, @mean, @std, inplace: @inplace)
      end
    end
  end
end
