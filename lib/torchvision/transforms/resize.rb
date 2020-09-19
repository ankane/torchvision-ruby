module TorchVision
  module Transforms
    class Resize
      def initialize(size)
        @size = size
      end

      def call(img)
        F.resize(img, @size)
      end
    end
  end
end
