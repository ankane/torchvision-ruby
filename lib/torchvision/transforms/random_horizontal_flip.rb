module TorchVision
  module Transforms
    class RandomHorizontalFlip < Torch::NN::Module
      def initialize(p: 0.5)
        super()
        @p = p
      end

      def forward(img)
        if Torch.rand(1).item < @p
          F.hflip(img)
        else
          img
        end
      end
    end
  end
end
