module TorchVision
  module Datasets
    class ImageFolder < DatasetFolder
      def initialize(root, transform: nil)
        super
        @imgs = @samples
      end
    end
  end
end
