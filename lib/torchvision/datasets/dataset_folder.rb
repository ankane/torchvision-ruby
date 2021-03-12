module TorchVision
  module Datasets
    class DatasetFolder < VisionDataset
      def initialize(root, transform: nil)
        super(root, transform: transform)
        # TODO
      end
    end
  end
end
