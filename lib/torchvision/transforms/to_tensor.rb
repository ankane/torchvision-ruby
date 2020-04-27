module TorchVision
  module Transforms
    class ToTensor
      def call(pic)
        F.to_tensor(pic)
      end
    end
  end
end
