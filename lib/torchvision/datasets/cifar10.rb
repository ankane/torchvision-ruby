module TorchVision
  module Datasets
    class CIFAR10 < VisionDataset
      def initialize(root, train: true, download: false, transform: nil, target_transform: nil)
        super(root, transform: transform, target_transform: target_transform)
        @train = train

        self.download if download

        if !check_exists
          raise Error, "Dataset not found. You can use download: true to download it"
        end
      end

      def download
        raise Torch::NotImplementedYet
      end
    end
  end
end
