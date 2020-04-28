module TorchVision
  module Datasets
    # TODO inherit Torch::Utils::Data::Dataset
    class VisionDataset
      def initialize(root, transforms: nil, transform: nil, target_transform: nil)
        @root = root

        has_transforms = !transforms.nil?
        has_separate_transform = !transform.nil? || !target_transform.nil?
        if has_transforms && has_separate_transform
          raise ArgumentError, "Only transforms or transform/target_transform can be passed as argument"
        end

        if has_separate_transform
          raise Torch::NotImplementedYet
          # transforms = StandardTransform.new(transform, target_transform)
        end
        @transforms = transforms
      end

      private

      def download_file(url, download_root:, filename:, sha256:)
        FileUtils.mkdir_p(download_root)

        dest = File.join(download_root, filename)
        return dest if File.exist?(dest)

        temp_path = "#{Dir.tmpdir}/#{Time.now.to_f}" # TODO better name

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
              end
            end
          end
        end

        unless check_integrity(temp_path, sha256)
          raise Error, "Bad hash"
        end

        FileUtils.mv(temp_path, dest)

        dest
      end

      def check_integrity(path, sha256)
        Digest::SHA256.file(path).hexdigest == sha256
      end
    end
  end
end
