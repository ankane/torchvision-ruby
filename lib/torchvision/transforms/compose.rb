module TorchVision
  module Transforms
    class Compose < Torch::NN::Module
      def initialize(transforms)
        @transforms = transforms
      end

      def forward(img)
        @transforms.each do |t|
          img = t.call(img)
        end
        img
      end
    end
  end
end
