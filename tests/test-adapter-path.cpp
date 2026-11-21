#include "adapter.h"

#include <chrono>
#include <fstream>

int main() {
    namespace fs = std::filesystem;
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path sandbox = fs::temp_directory_path() / ("mm3-adapter-path-" + std::to_string(stamp));
    const fs::path verbatim = fs::u8path(adapter_root(sandbox.u8string()));
    bool ok = false;
    try {
        const fs::path label = fs::u8path(u8"\u0433\u043e\u043b\u043e\u0441");
        fs::path root = sandbox / label;
        for (int index = 0; index < 5; ++index) {
            root /= std::string(54, static_cast<char>('a' + index));
        }
        const std::string ordinary = root.u8string();
        const std::string extended = adapter_root(ordinary);
        const fs::path adapter = fs::u8path(extended) / label;
        fs::create_directories(adapter);
        const std::string header =
            R"({"transformer.transformer_blocks.0.attn.to_q.lora_A.weight":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}})";
        {
            std::ofstream weights(adapter / "adapter_model.safetensors", std::ios::binary);
            const uint64_t size = header.size();
            const float factor = 1.0f;
            weights.write(reinterpret_cast<const char *>(&size), sizeof(size));
            weights.write(header.data(), header.size());
            weights.write(reinterpret_cast<const char *>(&factor), sizeof(factor));
        }
        std::ofstream(adapter / "adapter_config.json") << R"({"lora_alpha":16})";
        std::string from_ordinary;
        std::string from_extended;
        const auto found = adapter_scan(ordinary);
        ok = ordinary.size() > MAX_PATH &&
             adapter_resolve(ordinary, label.u8string(), &from_ordinary) &&
             adapter_resolve(extended, label.u8string(), &from_extended) &&
             from_ordinary == from_extended &&
             adapter_files(from_ordinary).size() == 1 &&
             adapter_read_config(from_ordinary).alpha == 16.0f &&
             found.size() == 1 && found.front().info.ok && found.front().info.dit_keys == 1 &&
             !adapter_resolve(ordinary, "..", &from_ordinary) &&
             adapter_root(R"(\\server\share\adapters)") == R"(\\?\UNC\server\share\adapters)";
    } catch (const std::exception & error) {
        fprintf(stderr, "%s\n", error.what());
    }
    std::error_code cleanup_error;
    fs::remove_all(verbatim, cleanup_error);
    if (!ok) {
        fprintf(stderr, "Windows adapter path regression failed\n");
        return 1;
    }
    return 0;
}
