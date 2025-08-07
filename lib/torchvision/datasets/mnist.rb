module TorchVision
  module Datasets
    class MNIST < VisionDataset
      # http://yann.lecun.com/exdb/mnist/
      def initialize(root, train: true, download: false, transform: nil, target_transform: nil)
        super(root, transform: transform, target_transform: target_transform)
        @train = train

        self.download if download

        if !check_exists
          raise Error, "Dataset not found. You can use download: true to download it"
        end

        data_file = @train ? training_file : test_file
        @data, @targets = Torch.load(File.join(processed_folder, data_file))
      end

      def size
        @data.size(0)
      end

      def [](index)
        img, target = @data[index], @targets[index].item

        img = Utils.image_from_array(img)

        img = @transform.call(img) if @transform

        target = @target_transform.call(target) if @target_transform

        [img, target]
      end

      def raw_folder
        File.join(@root, self.class.name.split("::").last, "raw")
      end

      def processed_folder
        File.join(@root, self.class.name.split("::").last, "processed")
      end

      def check_exists
        File.exist?(File.join(processed_folder, training_file)) &&
          File.exist?(File.join(processed_folder, test_file))
      end

      def download
        return if check_exists

        FileUtils.mkdir_p(raw_folder)
        FileUtils.mkdir_p(processed_folder)

        resources.each do |resource|
          success = false
          mirrors.each do |mirror|
            begin
              url = "#{mirror}#{resource[:filename]}"
              download_file(url, download_root: raw_folder, filename: resource[:filename], sha256: resource[:sha256])
              success = true
              break
            rescue Errno::ECONNREFUSED, Net::HTTPFatalError, Net::HTTPClientException => e
              puts "Failed to download (trying next): #{e.message}"
            end
          end
          raise Error, "Error downloading #{resource[:filename]}" unless success
        end

        puts "Processing..."

        training_set = [
          unpack_mnist("train-images-idx3-ubyte", 16, [60000, 28, 28]),
          unpack_mnist("train-labels-idx1-ubyte", 8, [60000])
        ]
        test_set = [
          unpack_mnist("t10k-images-idx3-ubyte", 16, [10000, 28, 28]),
          unpack_mnist("t10k-labels-idx1-ubyte", 8, [10000])
        ]

        Torch.save(training_set, File.join(processed_folder, training_file))
        Torch.save(test_set, File.join(processed_folder, test_file))

        puts "Done!"
      end

      private

      def mirrors
        [
          "https://ossci-datasets.s3.amazonaws.com/mnist/",
          "https://yann.lecun.com/exdb/mnist/"
        ]
      end

      def resources
        [
          {
            filename: "train-images-idx3-ubyte.gz",
            sha256: "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"
          },
          {
            filename: "train-labels-idx1-ubyte.gz",
            sha256: "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"
          },
          {
            filename: "t10k-images-idx3-ubyte.gz",
            sha256: "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"
          },
          {
            filename: "t10k-labels-idx1-ubyte.gz",
            sha256: "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"
          }
        ]
      end

      def training_file
        "training.pt"
      end

      def test_file
        "test.pt"
      end

      def unpack_mnist(path, offset, shape)
        path = File.join(raw_folder, "#{path}.gz")
        File.open(path, "rb") do |f|
          gz = Zlib::GzipReader.new(f)
          gz.read(offset)
          Torch.tensor(Numo::UInt8.from_string(gz.read, shape))
        end
      end
    end
  end
end
