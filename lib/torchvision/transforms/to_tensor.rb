module TorchVision
  module Transforms
    class ToTensor < Torch::NN::Module
      def forward(pic)
        F.to_tensor(pic)
      end
    end
  end
end
