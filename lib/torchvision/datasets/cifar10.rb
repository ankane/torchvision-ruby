module TorchVision
  module Datasets
    class CIFAR10 < VisionDataset
      # https://www.cs.toronto.edu/~kriz/cifar.html

      def initialize(root, train: true, download: false, transform: nil, target_transform: nil)
        super(root, transform: transform, target_transform: target_transform)
        @train = train

        self.download if download

        if !_check_integrity
          raise Error, "Dataset not found or corrupted. You can use download=True to download it"
        end

        downloaded_list = @train ? train_list : test_list

        @data = String.new
        @targets = String.new

        i = 0
        downloaded_list.each do |file|
          file_path = File.join(@root, base_folder, file[:filename])
          File.open(file_path, "rb") do |f|
            while !f.eof?
              @targets << f.read(1)
              @data << f.read(3072)
              i += 1
            end
          end
        end

        @targets = @targets.unpack("C*")
        # TODO switch i to -1 when Numo supports it
        @data = Numo::UInt8.from_binary(@data).reshape(i, 3, 32, 32)
        @data = @data.transpose(0, 2, 3, 1)
      end

      def size
        @data.shape[0]
      end

      def [](index)
        # TODO remove trues when Numo supports it
        img, target = @data[index, true, true, true], @targets[index]

        # TODO convert to image
        img = @transform.call(img) if @transform

        target = @target_transform.call(target) if @target_transform

        [img, target]
      end

      def _check_integrity
        root = @root
        (train_list + test_list).each do |fentry|
          fpath = File.join(root, base_folder, fentry[:filename])
          return false unless check_integrity(fpath, fentry[:sha256])
        end
        true
      end

      def download
        if _check_integrity
          puts "Files already downloaded and verified"
          return
        end

        download_file(url, download_root: @root, filename: filename, sha256: tgz_sha256)

        path = File.join(@root, filename)
        File.open(path, "rb") do |io|
          Gem::Package.new("").extract_tar_gz(io, @root)
        end
      end

      private

      def base_folder
        "cifar-10-batches-bin"
      end

      def url
        "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
      end

      def filename
        "cifar-10-binary.tar.gz"
      end

      def tgz_sha256
        "c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd"
      end

      def train_list
        [
          {filename: "data_batch_1.bin", sha256: "cee916563c9f80d84e3cc88e17fdc0941787f1244f00a67874d45b261883ada5"},
          {filename: "data_batch_2.bin", sha256: "a591ca11fa1708a91ee40f54b3da4784ccd871ecf2137de63f51ada8b3fa57ed"},
          {filename: "data_batch_3.bin", sha256: "bbe8596564c0f86427f876058170b84dac6670ddf06d79402899d93ceea26f67"},
          {filename: "data_batch_4.bin", sha256: "014e562d6e23c72197cc727519169a60359f5eccd8945ad5a09d710285ff4e48"},
          {filename: "data_batch_5.bin", sha256: "755304fc0b379caeae8c14f0dac912fbc7d6cd469eb67a1029a08a39453a9add"},
        ]
      end

      def test_list
        [
          {filename: "test_batch.bin", sha256: "8e2eb146ae340b09e24670f29cabc6326dba54da8789dab6768acf480273f65b"}
        ]
      end
    end
  end
end
