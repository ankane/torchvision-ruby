module TorchVision
  module Datasets
    class CIFAR10 < VisionDataset
      BASE_FOLDER = "cifar-10-batches-bin"
      URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
      FILENAME = "cifar-10-binary.tar.gz"
      SHA256 = "c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd"

      def initialize(root, train: true, download: false, transform: nil, target_transform: nil)
        super(root, transform: transform, target_transform: target_transform)
        @train = train

        self.download if download

        if !check_exists
          raise Error, "Dataset not found. You can use download: true to download it"
        end
      end

      def download
        download_file(URL, download_root: @root, filename: FILENAME, sha256: SHA256)

        raise Torch::NotImplementedYet
      end
    end
  end
end
