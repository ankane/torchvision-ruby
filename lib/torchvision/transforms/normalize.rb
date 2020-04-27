module TorchVision
  module Transforms
    class Normalize
      def initialize(mean, std, inplace: false)
        @mean = mean
        @std = std
        @inplace = inplace
      end

      def call(tensor)
        F.normalize(tensor, @mean, @std, inplace: @inplace)
      end
    end
  end
end
