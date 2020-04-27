module TorchVision
  module Transforms
    class Compose
      def initialize(transforms)
        @transforms = transforms
      end

      def call(img)
        @transforms.each do |t|
          img = t.call(img)
        end
        img
      end
    end
  end
end
