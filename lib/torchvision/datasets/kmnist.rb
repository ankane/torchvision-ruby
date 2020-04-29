module TorchVision
  module Datasets
    class KMNIST < MNIST
      # https://github.com/rois-codh/kmnist

      private

      def resources
        [
          {
            url: "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz",
            sha256: "51467d22d8cc72929e2a028a0428f2086b092bb31cfb79c69cc0a90ce135fde4"
          },
          {
            url: "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz",
            sha256: "e38f9ebcd0f3ebcdec7fc8eabdcdaef93bb0df8ea12bee65224341c8183d8e17"
          },
          {
            url: "http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz",
            sha256: "edd7a857845ad6bb1d0ba43fe7e794d164fe2dce499a1694695a792adfac43c5"
          },
          {
            url: "http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz",
            sha256: "20bb9a0ef54c7db3efc55a92eef5582c109615df22683c380526788f98e42a1c"
          }
        ]
      end
    end
  end
end
