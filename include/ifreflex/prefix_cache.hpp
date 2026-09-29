// LRU cache of decoded state prefixes, keyed by the exact prefix token ids.
//
// A state seen before skips its prefill: the decoded sequence state is
// checkpointed (KV cache for attention models, recurrent state for RWKV/hybrid,
// via llama_state_seq_get/set_data) and restored on the next request. Bounded by
// host RAM. Adapted from dohnuts.cpp include/dohnuts/prefix_cache.hpp (MIT).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace ifreflex {

// Prefixes shorter than this are cheaper to decode again than to checkpoint.
inline constexpr size_t PREFIX_MIN_TOKENS = 16;

// Total host RAM the cache may hold, in bytes (overridden by --prefix-cache-mib).
inline constexpr size_t PREFIX_CACHE_LIMIT = 256u * 1024 * 1024;

class prefix_cache {
public:
    explicit prefix_cache(size_t limit = PREFIX_CACHE_LIMIT) : limit(limit) {}

    // Returns the checkpoint for the exact prefix tokens, or nullptr. A hit moves
    // the entry to the front of the LRU.
    const std::vector<uint8_t> * get(const std::vector<int32_t> & ids) {
        const std::string key = encode(ids);
        auto it = index.find(key);
        if (it == index.end()) return nullptr;
        entries.splice(entries.begin(), entries, it->second);
        return &entries.front().state;
    }

    // Drops the entry for these tokens (e.g. after a failed restore).
    void drop(const std::vector<int32_t> & ids) {
        auto it = index.find(encode(ids));
        if (it == index.end()) return;
        bytes -= it->second->state.size();
        entries.erase(it->second);
        index.erase(it);
    }

    // Stores a checkpoint, evicting least-recently-used entries to stay under the
    // limit. Empty or oversized checkpoints are ignored.
    void put(const std::vector<int32_t> & ids, std::vector<uint8_t> state) {
        if (state.empty() || state.size() > limit) return;
        const std::string key = encode(ids);
        auto existing = index.find(key);
        if (existing != index.end()) {
            bytes -= existing->second->state.size();
            entries.erase(existing->second);
            index.erase(existing);
        }
        bytes += state.size();
        entries.push_front({key, std::move(state)});
        index[key] = entries.begin();
        while (bytes > limit && !entries.empty()) {
            bytes -= entries.back().state.size();
            index.erase(entries.back().key);
            entries.pop_back();
        }
    }

    size_t size_in_bytes() const { return bytes; }
    size_t count() const { return entries.size(); }

private:
    static std::string encode(const std::vector<int32_t> & ids) {
        std::string s;
        s.resize(ids.size() * sizeof(int32_t));
        if (!ids.empty()) std::memcpy(s.data(), ids.data(), s.size());
        return s;
    }

    struct entry {
        std::string key;
        std::vector<uint8_t> state;
    };
    std::list<entry> entries; // front = most recently used
    std::unordered_map<std::string, std::list<entry>::iterator> index;
    size_t bytes = 0;
    size_t limit;
};

} // namespace ifreflex
