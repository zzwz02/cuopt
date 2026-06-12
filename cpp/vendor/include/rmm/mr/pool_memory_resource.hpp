/*
 * cuOpt vendored RMM shim — stream-ordered arena (sub-allocating) pool resource.
 *
 * Faithful to rmm::mr::pool_memory_resource semantics, which the thin
 * cudaMallocFromPoolAsync wrapper this replaces was NOT:
 *  - Memory comes from large upstream slabs (cudaMalloc via Upstream) that are
 *    sub-allocated and NEVER freed back until destruction, so freed virtual
 *    addresses stay mapped. This (a) keeps pointers captured inside replayed
 *    CUDA graphs stable and (b) masks latent stale-pointer reads in consumer
 *    code exactly like rmm's arena does (the driver mempool unmaps freed VAs,
 *    which is what made ROUTING_UNIT_TEST's latent use-after-free segfault).
 *  - Stream-ordered reuse: each free list is keyed by a per-stream event
 *    (thread-local for the per-thread default stream, matching rmm). A block
 *    freed and re-allocated on the same stream needs no synchronization;
 *    taking blocks freed on another stream first makes the allocating stream
 *    cudaStreamWaitEvent on the freeing stream's event, then merges that whole
 *    list (rmm's get_block_from_other_stream).
 *  - Adjacent free blocks coalesce within a list (rmm's coalescing_free_list).
 *
 * Divergences (deliberate, documented):
 *  - New slabs are cudaMemset to zero once at creation ("first-touch zeroed",
 *    like a fresh arena); recycled blocks are NOT zeroed — same as rmm.
 *  - Default initial size when unspecified is 256 MiB (rmm uses half of free
 *    device memory, which over-commits when several test processes share a GPU).
 *  - Thread-local events for the per-thread default stream are intentionally
 *    leaked (created once per thread per resource); resources here are
 *    process-lifetime singletons set via set_per_device_resource.
 *  - Growth uses synchronous cudaMalloc and is therefore illegal during stream
 *    capture — identical to rmm, whose pool also grows via the upstream
 *    cudaMalloc. Size the initial pool so capture-phase growth does not occur.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rmm::mr {

template <typename Upstream = device_memory_resource>
class pool_memory_resource final : public device_memory_resource {
 public:
  explicit pool_memory_resource(Upstream* upstream,
                                std::optional<std::size_t> initial_pool_size = {},
                                std::optional<std::size_t> maximum_pool_size = {})
    : upstream_{upstream},
      device_id_{get_current_cuda_device()},
      initial_size_{round_up(initial_pool_size.value_or(default_initial_size))},
      max_size_{maximum_pool_size}
  {
    RMM_EXPECTS(upstream_ != nullptr, "Unexpected null upstream resource.");
  }

  explicit pool_memory_resource(Upstream& upstream,
                                std::optional<std::size_t> initial_pool_size = {},
                                std::optional<std::size_t> maximum_pool_size = {})
    : pool_memory_resource{&upstream, initial_pool_size, maximum_pool_size}
  {
  }

  pool_memory_resource(pool_memory_resource const&)            = delete;
  pool_memory_resource& operator=(pool_memory_resource const&) = delete;
  pool_memory_resource(pool_memory_resource&&)                 = delete;
  pool_memory_resource& operator=(pool_memory_resource&&)      = delete;

  ~pool_memory_resource() override
  {
    // Tolerate CUDA teardown ordering: never throw from the destructor.
    for (auto& [stream, event] : stream_events_) {
      if (event != nullptr) { cudaEventDestroy(event); }
    }
    for (auto& [ptr, size] : slabs_) {
      try {
        upstream_->deallocate(ptr, size, cuda_stream_view{});
      } catch (...) {  // NOLINT(bugprone-empty-catch)
      }
    }
    cudaGetLastError();
  }

  [[nodiscard]] std::size_t pool_size() const noexcept { return capacity_; }

 private:
  static constexpr std::size_t pool_alignment       = CUDA_ALLOCATION_ALIGNMENT;  // 256
  static constexpr std::size_t default_initial_size = std::size_t{256} << 20;     // 256 MiB

  [[nodiscard]] static std::size_t round_up(std::size_t bytes) noexcept
  {
    return align_up(bytes, pool_alignment);
  }

  // Free blocks owned by one freeing stream, keyed externally by that stream's
  // event. Address-ordered so adjacent blocks coalesce on insert.
  struct stream_list_t {
    cudaStream_t stream{};
    std::map<char*, std::size_t> blocks;  // ptr -> size
  };

  // One event per freeing stream. The per-thread default stream shares a
  // handle value across threads but designates DIFFERENT streams, so it gets a
  // thread-local event per resource (rmm does the same): blocks freed by
  // another thread's PTDS land in a different list and are only reused after a
  // cudaStreamWaitEvent on that thread's event.
  cudaEvent_t event_for(cuda_stream_view stream)
  {
    if (stream.is_per_thread_default()) {
      static thread_local std::unordered_map<pool_memory_resource const*, cudaEvent_t> tl_events;
      auto [it, inserted] = tl_events.try_emplace(this, nullptr);
      if (inserted) {
        RMM_CUDA_TRY(cudaEventCreateWithFlags(&it->second, cudaEventDisableTiming));
      }
      return it->second;
    }
    auto [it, inserted] = stream_events_.try_emplace(stream.value(), nullptr);
    if (inserted) { RMM_CUDA_TRY(cudaEventCreateWithFlags(&it->second, cudaEventDisableTiming)); }
    return it->second;
  }

  static void insert_coalesced(stream_list_t& list, char* ptr, std::size_t size)
  {
    auto next = list.blocks.lower_bound(ptr);
    if (next != list.blocks.begin()) {
      auto prev = std::prev(next);
      if (prev->first + prev->second == ptr) {  // merge with predecessor
        ptr  = prev->first;
        size += prev->second;
        list.blocks.erase(prev);
      }
    }
    if (next != list.blocks.end() && ptr + size == next->first) {  // merge with successor
      size += next->second;
      list.blocks.erase(next);
    }
    list.blocks.emplace(ptr, size);
  }

  [[nodiscard]] static bool has_fit(stream_list_t const& list, std::size_t size) noexcept
  {
    for (auto const& [ptr, block_size] : list.blocks) {
      if (block_size >= size) { return true; }
    }
    return false;
  }

  // Best-fit take; splits the block, leaving the tail in the list. Blocks are
  // taken/split only at pool_alignment multiples (slabs are 256-aligned and all
  // sizes are rounded), so every returned pointer is 256-aligned.
  static char* try_take(stream_list_t& list, std::size_t size)
  {
    auto best = list.blocks.end();
    for (auto it = list.blocks.begin(); it != list.blocks.end(); ++it) {
      if (it->second >= size && (best == list.blocks.end() || it->second < best->second)) {
        best = it;
      }
    }
    if (best == list.blocks.end()) { return nullptr; }
    char* ptr                  = best->first;
    std::size_t const old_size = best->second;
    list.blocks.erase(best);
    if (old_size > size) {
      // Tail of a hole: cannot be adjacent to another free block, plain insert.
      list.blocks.emplace(ptr + size, old_size - size);
    }
    return ptr;
  }

  // Merge `from` into `into`, coalescing. Caller has made `into`'s stream wait
  // on `from`'s event first.
  static void merge_lists(stream_list_t& into, stream_list_t& from)
  {
    for (auto const& [ptr, size] : from.blocks) {
      insert_coalesced(into, ptr, size);
    }
    from.blocks.clear();
  }

  // Grow the arena with a new upstream slab. Synchronous; illegal during
  // stream capture (same constraint as rmm's pool).
  void grow(std::size_t bytes, stream_list_t& into)
  {
    std::size_t want = (capacity_ == 0) ? std::max(bytes, initial_size_)
                                        : std::max(bytes, capacity_);  // double total
    if (max_size_.has_value()) {
      RMM_EXPECTS(capacity_ + bytes <= *max_size_,
                  "Maximum pool size exceeded",
                  rmm::out_of_memory);
      want = std::min(want, *max_size_ - capacity_);
    }
    void* slab = nullptr;
    try {
      slab = upstream_->allocate(want, cuda_stream_view{});
    } catch (rmm::bad_alloc const&) {
      cudaGetLastError();  // clear sticky error before retrying
      want = bytes;        // fall back to exactly what is needed
      slab = upstream_->allocate(want, cuda_stream_view{});
    }
    // First-touch zero, like a fresh arena; recycled blocks are never re-zeroed.
    RMM_CUDA_TRY(cudaMemset(slab, 0, want));
    RMM_CUDA_TRY(cudaDeviceSynchronize());
    slabs_.emplace_back(slab, want);
    capacity_ += want;
    insert_coalesced(into, static_cast<char*>(slab), want);
  }

  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
  {
    if (bytes == 0) { return nullptr; }
    bytes = round_up(bytes);
    cuda_set_device_raii set_dev{device_id_};
    std::lock_guard<std::mutex> lock{mtx_};

    cudaEvent_t const ev = event_for(stream);
    auto& mine           = lists_[ev];
    mine.stream          = stream.value();

    // 1) Same-stream reuse: stream-ordered, no synchronization needed.
    if (char* ptr = try_take(mine, bytes)) { return ptr; }

    // 2) Steal from a single other stream's list that can satisfy the request.
    for (auto it = lists_.begin(); it != lists_.end(); ++it) {
      if (it->first == ev || it->second.blocks.empty()) { continue; }
      if (has_fit(it->second, bytes)) {
        RMM_CUDA_TRY(cudaStreamWaitEvent(stream.value(), it->first, 0));
        merge_lists(mine, it->second);
        lists_.erase(it);
        return try_take(mine, bytes);  // guaranteed: the fitting block is now ours
      }
    }

    // 3) Fragmentation rescue: merge ALL lists (waiting on each event) — blocks
    //    adjacent across lists may coalesce into a large-enough one.
    bool merged_any = false;
    for (auto it = lists_.begin(); it != lists_.end();) {
      if (it->first == ev || it->second.blocks.empty()) {
        ++it;
        continue;
      }
      RMM_CUDA_TRY(cudaStreamWaitEvent(stream.value(), it->first, 0));
      merge_lists(mine, it->second);
      it         = lists_.erase(it);
      merged_any = true;
    }
    if (merged_any) {
      if (char* ptr = try_take(mine, bytes)) { return ptr; }
    }

    // 4) Grow the arena.
    grow(bytes, mine);
    char* ptr = try_take(mine, bytes);
    RMM_EXPECTS(ptr != nullptr, "Pool allocation failed after growth", rmm::out_of_memory);
    return ptr;
  }

  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override
  {
    if (ptr == nullptr || bytes == 0) { return; }
    bytes = round_up(bytes);
    cuda_set_device_raii set_dev{device_id_};
    std::lock_guard<std::mutex> lock{mtx_};

    cudaEvent_t const ev = event_for(stream);
    auto& list           = lists_[ev];
    list.stream          = stream.value();
    insert_coalesced(list, static_cast<char*>(ptr), bytes);
    // Record AFTER posting the free: all prior work on `stream` that used `ptr`
    // happens-before this event; other streams reuse only after waiting on it.
    auto const status = cudaEventRecord(ev, stream.value());
    if (status != cudaSuccess && status != cudaErrorCudartUnloading) { cudaGetLastError(); }
  }

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    return this == &other;
  }

  Upstream* upstream_;
  cuda_device_id device_id_;
  std::size_t initial_size_;
  std::optional<std::size_t> max_size_;

  std::mutex mtx_;
  std::vector<std::pair<void*, std::size_t>> slabs_;
  std::size_t capacity_{0};
  std::unordered_map<cudaStream_t, cudaEvent_t> stream_events_;  // non-PTDS streams
  std::unordered_map<cudaEvent_t, stream_list_t> lists_;
};

// Deduction guides for `pool_memory_resource(upstream, size)` style construction.
template <typename Upstream>
pool_memory_resource(Upstream*, std::optional<std::size_t> = {}, std::optional<std::size_t> = {})
  -> pool_memory_resource<Upstream>;
template <typename Upstream>
pool_memory_resource(Upstream&, std::optional<std::size_t> = {}, std::optional<std::size_t> = {})
  -> pool_memory_resource<Upstream>;

}  // namespace rmm::mr
