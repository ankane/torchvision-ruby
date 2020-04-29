module TorchVision
  module Datasets
    class CIFAR100 < CIFAR10
      # https://www.cs.toronto.edu/~kriz/cifar.html

      private

      def base_folder
        "cifar-100-binary"
      end

      def url
        "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
      end

      def filename
        "cifar-100-binary.tar.gz"
      end

      def tgz_sha256
        "58a81ae192c23a4be8b1804d68e518ed807d710a4eb253b1f2a199162a40d8ec"
      end

      def train_list
        [
          {filename: "train.bin", sha256: "f31298fc616915fa142368359df1c4ca2ae984d6915ca468b998a5ec6aeebf29"}
        ]
      end

      def test_list
        [
          {filename: "test.bin", sha256: "d8b1e6b7b3bee4020055f0699b111f60b1af1e262aeb93a0b659061746f8224a"}
        ]
      end
    end
  end
end
