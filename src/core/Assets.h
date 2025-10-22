#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace core {

class AssetRegistry {
public:
    struct Entry {
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
        std::uint64_t hash = 0;
    };

    bool initialize(const std::string& assetDir);
    bool initialize(const std::string& pakPath, const std::string& indexPath);

    bool contains(std::string_view path) const;
    bool getEntry(std::string_view path, Entry& out) const;

    std::span<const std::byte> view(std::string_view path) const;
    bool loadBinary(std::string_view path, std::vector<std::uint8_t>& out) const;
    bool loadText(std::string_view path, std::string& out) const;

    const std::vector<std::string>& assetList() const { return assetOrder_; }

private:
    bool readPak(const std::string& pakPath);
    bool readIndex(const std::string& indexPath);

    std::vector<std::byte> pakData_;
    std::unordered_map<std::string, Entry> entries_;
    std::vector<std::string> assetOrder_;
};

} // namespace core
