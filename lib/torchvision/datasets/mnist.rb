module TorchVision
  module Datasets
    class MNIST
      RESOURCES = [
        {
          url: "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
          sha256: "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"
        },
        {
          url: "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
          sha256: "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"
        },
        {
          url: "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
          sha256: "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"
        },
        {
          url: "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
          sha256: "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"
        }
      ]
      TRAINING_FILE = "training.pt"
      TEST_FILE = "test.pt"

      def initialize(root, train: true, download: false, transform: nil)
        @root = root
        @train = train
        @transform = transform

        self.download if download

        if !check_exists
          raise Error, "Dataset not found. You can use download: true to download it"
        end

        data_file = @train ? TRAINING_FILE : TEST_FILE
        @data, @targets = Torch.load(File.join(processed_folder, data_file))
      end

      def size
        @data.size[0]
      end

      def [](index)
        img = @data[index]
        img = @transform.call(img) if @transform

        target = @targets[index].item

        [img, target]
      end

      def raw_folder
        File.join(@root, "MNIST", "raw")
      end

      def processed_folder
        File.join(@root, "MNIST", "processed")
      end

      def check_exists
        File.exist?(File.join(processed_folder, TRAINING_FILE)) &&
          File.exist?(File.join(processed_folder, TEST_FILE))
      end

      def download
        return if check_exists

        FileUtils.mkdir_p(raw_folder)
        FileUtils.mkdir_p(processed_folder)

        RESOURCES.each do |resource|
          filename = resource[:url].split("/").last
          download_file(resource[:url], download_root: raw_folder, filename: filename, sha256: resource[:sha256])
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

        Torch.save(training_set, File.join(processed_folder, TRAINING_FILE))
        Torch.save(test_set, File.join(processed_folder, TEST_FILE))

        puts "Done!"
      end

      private

      def unpack_mnist(path, offset, shape)
        path = File.join(raw_folder, "#{path}.gz")
        File.open(path, "rb") do |f|
          gz = Zlib::GzipReader.new(f)
          gz.read(offset)
          Torch.tensor(Numo::UInt8.from_string(gz.read, shape))
        end
      end

      def download_file(url, download_root:, filename:, sha256:)
        FileUtils.mkdir_p(download_root)

        dest = File.join(download_root, filename)
        return dest if File.exist?(dest)

        temp_path = "#{Dir.tmpdir}/#{Time.now.to_f}" # TODO better name

        digest = Digest::SHA256.new

        uri = URI(url)

        # Net::HTTP automatically adds Accept-Encoding for compression
        # of response bodies and automatically decompresses gzip
        # and deflateresponses unless a Range header was sent.
        # https://ruby-doc.org/stdlib-2.6.4/libdoc/net/http/rdoc/Net/HTTP.html
        Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https") do |http|
          request = Net::HTTP::Get.new(uri)

          puts "Downloading #{url}..."
          File.open(temp_path, "wb") do |f|
            http.request(request) do |response|
              response.read_body do |chunk|
                f.write(chunk)
                digest.update(chunk)
              end
            end
          end
        end

        if digest.hexdigest != sha256
          raise Error, "Bad hash: #{digest.hexdigest}"
        end

        FileUtils.mv(temp_path, dest)

        dest
      end
    end
  end
end
