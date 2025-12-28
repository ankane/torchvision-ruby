require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"

class Minitest::Test
  def root
    @root ||= ENV["CI"] ? "#{ENV["HOME"]}/data" : Dir.tmpdir
  end
end
