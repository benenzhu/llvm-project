
#include <memory>
#include <string>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>
#include <type_traits>
#include <concepts>
#include <memory>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <pybind11/pybind11.h>
#include <iostream>
namespace ____start{


};
# 1 "/root/HipKittens//include/kittens.cuh" 1
/**
 * @file
 * @brief The master header file of ThunderKittens. This file includes everything you need!
 */



# 1 "/root/HipKittens//include/common/common.cuh" 1
/**
 * @file
 * @brief A collection of common resources on which ThunderKittens depends.
 */




# 1 "/root/HipKittens//include/common/util.cuh" 1
/**
 * @file
 * @brief General utilities for ThunderKittens.
 */
# 15 "/root/HipKittens//include/common/util.cuh"
# 1 "/root/HipKittens//include/common/base_types.cuh" 1
/**
 * @file
 * @brief Declarations, manipulations, and wrappers for basic types.
 * 
 * This file is a bunch of utilities for going back and forth between different types.
 * 
 * Many of them are for the compiler, so as to clean up the code. It unfortunately
 * seems necessary when we have types we really care about that are less than word width.
 */
# 19 "/root/HipKittens//include/common/base_types.cuh"
namespace kittens {

// /**
//  * @brief Bfloat16 floating-point type.
//  */
using bf16 = __hip_bfloat16;
/**
 * @brief Half-precision floating-point type.
 */
using half = __half;
// /**
//  * @brief Packed word of two bfloat16 floating-point values.
//  */
using bf16_2 = __hip_bfloat162;
/**
 * @brief Packed word of two half-precision floating-point values.
 */
using half_2 = __half2;

namespace ducks {
/**
 * @namespace base_types
 *
 * @brief A namespace for concepts for basic data types.
 */
namespace base_types {

template<typename T>
concept T2 = std::is_same_v<T, float2> || std::is_same_v<T, bf16_2> || std::is_same_v<T, half_2>;
template<typename T>
concept T1 = std::is_same_v<T, float> || std::is_same_v<T, bf16 > || std::is_same_v<T, half>;

} // namespace base_types
} // namespace ducks

/**
 * @namespace base_types
 *
 * @brief A namespace for ThunderKittens basic data types.
 */
namespace base_types {

/**
 * @brief Provides compile-time constants for different types.
 *
 * @tparam T The type for which to provide constants.
 */
template<typename T> struct constants {
    /**
     * @brief Zero
     * @return Constexpr zero with type T
     */
    static __attribute__((device)) inline constexpr T zero() { return T{0}; }
    /**
     * @brief One
     * @return Constexpr one with type T
     */
    static __attribute__((device)) inline constexpr T one() { return T{1}; }
    /**
     * @brief Positive infinity. Particularly useful for initializing before a min op.
     * @return Constexpr positive infinity with type T
     */
    static __attribute__((device)) inline constexpr T pos_infty() { return T{(__builtin_inff ())}; } // I'll find a better way at some point but this appears to work.
    /**
     * @brief Negative infinity. Particularly useful for initializing before a max op.
     * @return Constexpr negative infinity with type T
     */
    static __attribute__((device)) inline constexpr T neg_infty() { return T{-(__builtin_inff ())}; }
};
template<> struct constants<float2> {
    static __attribute__((device)) inline constexpr float2 zero() { return float2{0.f, 0.f}; }
    static __attribute__((device)) inline constexpr float2 one() { return float2{1.f, 1.f}; }
    static __attribute__((device)) inline constexpr float2 pos_infty() { return float2{constants<float>::pos_infty(), constants<float>::pos_infty()}; }
    static __attribute__((device)) inline constexpr float2 neg_infty() { return float2{constants<float>::neg_infty(), constants<float>::neg_infty()}; }
};
template<> struct constants<bf16> {
    static __attribute__((device)) inline constexpr bf16 zero() { return std::bit_cast<bf16>(uint16_t(0x0000)); } // unfortunately __float2bf16_rn is not constexpr
    static __attribute__((device)) inline constexpr bf16 one() { return std::bit_cast<bf16>(uint16_t(0x3F80)); }
    static __attribute__((device)) inline constexpr bf16 pos_infty() { return std::bit_cast<bf16>(uint16_t(0x7F80)); }
    static __attribute__((device)) inline constexpr bf16 neg_infty() { return std::bit_cast<bf16>(uint16_t(0xFF80)); }
};
template<> struct constants<bf16_2> {
    static __attribute__((device)) inline bf16_2 zero() { return bf16_2{constants<bf16>::zero(), constants<bf16>::zero()}; }
    static __attribute__((device)) inline bf16_2 one() { return bf16_2{constants<bf16>::one(), constants<bf16>::one()}; }
    static __attribute__((device)) inline bf16_2 pos_infty() { return bf16_2{constants<bf16>::pos_infty(), constants<bf16>::pos_infty()}; }
    static __attribute__((device)) inline bf16_2 neg_infty() { return bf16_2{constants<bf16>::neg_infty(), constants<bf16>::neg_infty()}; }
};
template<> struct constants<half> {
    static __attribute__((device)) inline constexpr half zero() { return std::bit_cast<half>(uint16_t(0x0000)); }
    static __attribute__((device)) inline constexpr half one() { return std::bit_cast<half>(uint16_t(0x3C00)); }
    static __attribute__((device)) inline constexpr half pos_infty() { return std::bit_cast<half>(uint16_t(0x7C00)); }
    static __attribute__((device)) inline constexpr half neg_infty() { return std::bit_cast<half>(uint16_t(0xFC00)); }
};
template<> struct constants<half_2> {
    static __attribute__((device)) inline constexpr half_2 zero() { return std::bit_cast<half_2>(uint32_t(0x00000000)); }
    static __attribute__((device)) inline constexpr half_2 one() { return std::bit_cast<half_2>(uint32_t(0x3C003C00)); }
    static __attribute__((device)) inline constexpr half_2 pos_infty() { return std::bit_cast<half_2>(uint32_t(0x7C007C00)); }
    static __attribute__((device)) inline constexpr half_2 neg_infty() { return std::bit_cast<half_2>(uint32_t(0xFC00FC00)); }
};
template<> struct constants<int> {
    static __attribute__((device)) inline constexpr int zero() { return 0; }
    static __attribute__((device)) inline constexpr int one() { return 1; }
};
template<> struct constants<int2> {
    static __attribute__((device)) inline constexpr int2 zero() { return int2{0, 0}; }
    static __attribute__((device)) inline constexpr int2 one() { return int2{1, 1}; }
};

/**
 * @brief Provides information about packing of elements for a given type.
 *
 * @tparam T The type for which to provide packing information.
 */
template<typename T> struct packing {
    /**
     * @brief The number of elements packed together.
     *
     * @return constexpr int representing number of elements within the type.
     */
    static __attribute__((device)) inline constexpr int num() { return 1; }
    /**
     * @brief Packs a single T element twice (replicated) into its packed type.
     *
     * @param i[in] The element to pack.
     * @return The packed type.
     */
    static __attribute__((device)) inline constexpr T pack(const auto &i);
};
template<> struct packing<bf16> {
    static __attribute__((device)) inline constexpr int num() { return 1; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    static __attribute__((device)) inline bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; }
};
template<> struct packing<bf16_2> {
    static __attribute__((device)) inline constexpr int num() { return 2; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    static __attribute__((device)) inline bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<half> {
    static __attribute__((device)) inline constexpr int num() { return 1; }
    using unpacked_type = half;
    using packed_type = half_2;
    static __attribute__((device)) inline constexpr half_2 pack(const half &i) { return std::bit_cast<half_2>(static_cast<uint32_t>(i) << 16 | static_cast<uint32_t>(i)); }
};
template<> struct packing<half_2> {
    static __attribute__((device)) inline constexpr int num() { return 2; }
    using unpacked_type = half;
    using packed_type = half_2;
    static __attribute__((device)) inline constexpr half_2 pack(const half &i) { return std::bit_cast<half_2>(static_cast<uint32_t>(i) << 16 | static_cast<uint32_t>(i)); } // this replication makes code cleaner later.
};
template<> struct packing<float> {
    static __attribute__((device)) inline constexpr int num() { return 1; }
    using unpacked_type = float;
    using packed_type = float2;
    static __attribute__((device)) inline constexpr float2 pack(const float &i) { return float2{i, i}; }
};
template<> struct packing<float2> {
    static __attribute__((device)) inline constexpr int num() { return 2; }
    using unpacked_type = float;
    using packed_type = float2;
    static __attribute__((device)) inline constexpr float2 pack(const float &i) { return float2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<int> {
    static __attribute__((device)) inline constexpr int num() { return 1; }
    using unpacked_type = int;
    using packed_type = int2;
    static __attribute__((device)) inline constexpr int2 pack(const int &i) { return int2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<int2> {
    static __attribute__((device)) inline constexpr int num() { return 2; }
    using unpacked_type = int;
    using packed_type = int2;
    static __attribute__((device)) inline constexpr int2 pack(const int &i) { return int2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<float4> {
    static __attribute__((device)) inline constexpr int num() { return 4; }
};
template<> struct packing<int4> {
    static __attribute__((device)) inline constexpr int num() { return 4; }
};

/**
 * @brief Provides templated functionality to convert between different types.
 *
 * @tparam T The target type for conversion.
 * @tparam U The source type for conversion.
 */
template<typename T, typename U> struct convertor {
    /**
     * @brief Converts a value of type U to type T.
     *
     * @param u[in] The value of type U to convert.
     * @return T The converted value of type T.
     */
    static __attribute__((host)) __attribute__((device)) inline T convert(const U & u) {
        return (T)u;
    }
};
template<> struct convertor<float, bf16> {
    static __attribute__((host)) __attribute__((device)) inline float convert(const bf16 & u) {
        return __bfloat162float(u);
    }
};
// template<> struct convertor<bf16, float> {
//     static __host__ __device__ inline bf16 convert(const float & u) {
//         return 	__float2bfloat16(u);
//     }
// };
template<> struct convertor<bf16, float> {
    static __attribute__((host)) __attribute__((device)) inline bf16 convert(const float &u) {
        // Fast unsafe conversion (truncation only)
        return std::bit_cast<bf16>(
            static_cast<uint16_t>(
                std::bit_cast<uint32_t>(u) >> 16
            )
        );
    }
};
template<> struct convertor<float2, bf16_2> {
    static __attribute__((host)) __attribute__((device)) inline float2 convert(const bf16_2 & u) {
        return __bfloat1622float2(u);
    }
};
template<> struct convertor<bf16_2, float2> {
    static __attribute__((host)) __attribute__((device)) inline bf16_2 convert(const float2 &u) {
        return bf16_2{
            std::bit_cast<bf16>(static_cast<uint16_t>(std::bit_cast<uint32_t>(u.x) >> 16)),
            std::bit_cast<bf16>(static_cast<uint16_t>(std::bit_cast<uint32_t>(u.y) >> 16))
        };
    }
};
// template<> struct convertor<bf16_2, float2> {
//     static __host__ __device__ inline bf16_2 convert(const float2 &u) {
//         uint32_t result;
//         asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" 
//                      : "=v"(result) 
//                      : "v"(u.x), "v"(u.y));
//         return *reinterpret_cast<bf16_2*>(&result);
//     }
// };


template<> struct convertor<float, half> {
    static __attribute__((host)) __attribute__((device)) inline float convert(const half & u) {
        return __half2float(u);
    }
};
template<> struct convertor<half, float> {
    static __attribute__((host)) __attribute__((device)) inline half convert(const float & u) {
        return __float2half(u);
    }
};
template<> struct convertor<float2, half_2> {
    static __attribute__((host)) __attribute__((device)) inline float2 convert(const half_2 & u) {
        return __half22float2(u);
    }
};
template<> struct convertor<half_2, float2> {
    static __attribute__((host)) __attribute__((device)) inline half_2 convert(const float2 & u) {
        return __float22half2_rn(u);
    }
};
template<> struct convertor<bf16, half> {
    static __attribute__((host)) __attribute__((device)) inline bf16 convert(const half & u) {
        return __float2bfloat16(__half2float(u));
    }
};
template<> struct convertor<half, bf16> {
    static __attribute__((host)) __attribute__((device)) inline half convert(const bf16 & u) {
        return __float2half(__bfloat162float(u));
    }
};
template<> struct convertor<bf16_2, half_2> {
    static __attribute__((host)) __attribute__((device)) inline bf16_2 convert(const half_2 & u) {
        return __float22bfloat162_rn(__half22float2(u));
    }
};
template<> struct convertor<half_2, bf16_2> {
    static __attribute__((host)) __attribute__((device)) inline half_2 convert(const bf16_2 & u) {
        return __float22half2_rn(__bfloat1622float2(u));
    }
};
}
}
# 16 "/root/HipKittens//include/common/util.cuh" 2





/**
 * @namespace kittens
 *
 * @brief The main namespace of ThunderKittens.
 */
namespace kittens {

/* ----------  GENERAL CONSTANTS FOR KITTENS  ---------- */

/**
 * @brief Tile dimension constant.
 */
template<typename T> constexpr int TILE_COL_DIM = sizeof(T) == 1 ? 32 : 16;
template<typename T> constexpr int TILE_ROW_DIM = 16;

/**
 * @brief Tile num elements constant calculated as TILE_DIM squared.
 */
template<typename T> constexpr int TILE_ELEMENTS{TILE_COL_DIM<T>*TILE_ROW_DIM<T>};
/**
 * @brief Constant representing number of threads in a warp.
 */
constexpr int WARP_THREADS{64};
/**
 * @brief Constant representing number of threads in a warpgroup of four warps.
 */
constexpr int WARPGROUP_THREADS{256};
/**

 * @brief Constant representing number of warps in a warpgroup of four warps.
 */
constexpr int WARPGROUP_WARPS{4};
/**

 * @brief Get the warp ID of the current thread.
 * @return The warp ID.
 */
__attribute__((device)) inline __attribute__((always_inline)) int warpid() { return threadIdx.x >> 6; }
/**
 * @brief Get the warpgroup ID of the current thread.
 * @return The warpgroup ID.
 */
__attribute__((device)) inline __attribute__((always_inline)) int warpgroupid() { return threadIdx.x >> 8; }

/**
 * @brief Get the lane ID of the current thread within its warp.
 * @return The lane ID.
 */
__attribute__((device)) inline __attribute__((always_inline)) int laneid() { return threadIdx.x & 0x3f; }


/**
 * @brief Compute the ceiling division of a by b.
 * @param a The dividend.
 * @param b The divisor.
 * @return The ceiling division of a by b.
 */
__attribute__((host)) __attribute__((device)) inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
  }

  /**
   * @brief Transform a workgroup ID to a new workgroup ID based on the chunk size and number of XCDs.
   * @param workgroup_id The original workgroup ID.
   * @param num_workgroups The total number of workgroups.
   * @param num_xcds The number of XCDs.
   * @param chunk_size The chunk size.
   * @return The new workgroup ID.
   */
  __attribute__((host)) __attribute__((device)) inline int chiplet_transform_chunked(
      int workgroup_id,
      int num_workgroups,
      int num_xcds,
      int chunk_size
  ) {
      // Current XCD
      int xcd = workgroup_id % num_xcds;

      // Largest full (NUM_XCDS*CHUNK_SIZE)-aligned block
      int block = num_xcds * chunk_size;
      int limit = (num_workgroups / block) * block;

      // If pid beyond the last full block, leave unchanged
      if (workgroup_id > limit) return workgroup_id;

      // Local PID (within round-robin assignment)
      int local_pid = workgroup_id / num_xcds;
      int chunk_idx = local_pid / chunk_size;
      int pos_in_chunk = local_pid % chunk_size;

      // New PID
      return chunk_idx * block + xcd * chunk_size + pos_in_chunk;
  }




constexpr int MAX_SHARED_MEMORY = 65536;
constexpr int NUM_XCDS = 8;
constexpr int CUS_PER_XCD = 38;
constexpr int NUM_CUS = CUS_PER_XCD * NUM_XCDS;







/* ----------  CUSTOM TYPES  ---------- */
typedef uint32_t uint2_t __attribute__((ext_vector_type(2)));

/* ----------  TYPE HELPERS  ---------- */

/**
 * @namespace ducks
 *
 * @brief ThunderKittens' namespace for template metaprogramming..
 * 
 * This includes primarily dummy types and concept wrappers, along
 * with a few additional utilities.
 */
namespace ducks {

/**
 * @brief A type representing an empty default for a template.
 */
struct default_type {};

// This macro can't be done as a template, so it doesn't really have a location in kittens.


}

/* ----------  SHUFFLE UTILS  ---------- */

/**
 * @brief Mask constant for all active threads in a warp.
 */
static constexpr uint64_t MASK_ALL = 0xFFFFFFFFFFFFFFFF;

/**
 * @brief Perform a shuffle down operation on a packed type synchronously across a warp.
 * @tparam T The type of the value to be shuffled.
 * @param mask[in] The mask of active threads.
 * @param f[in] The value to be shuffled.
 * @param delta[in] The number of positions to shuffle down.
 * @return The result of the shuffle operation.
 */
 template<typename T>
 __attribute__((device)) static inline T packed_shfl_down(uint64_t mask, const T &f, int delta) {
     return __shfl_down(f, delta, 64);
 }
template<>
__attribute__((device)) inline float2 packed_shfl_down<float2>(uint64_t mask, const float2 &f, int delta) {
    float2 r;
    r.x = __shfl_down(f.x, delta, 64); // Add the width parameter here
    r.y = __shfl_down(f.y, delta, 64); // And here
    return r;
}
template<>
__attribute__((device)) inline bf16 packed_shfl_down(uint64_t mask, const bf16 &f, int delta) {
    float r = __shfl_down(base_types::convertor<float, bf16>::convert(f), delta, 64);
    return base_types::convertor<bf16, float>::convert(r);
}

template<>
__attribute__((device)) inline bf16_2 packed_shfl_down(uint64_t mask, const bf16_2 &f, int delta) {
    float2 r;
    r.x = __shfl_down(base_types::convertor<float, bf16>::convert(f.x), delta, 64);
    r.y = __shfl_down(base_types::convertor<float, bf16>::convert(f.y), delta, 64);
    return base_types::convertor<bf16_2, float2>::convert(r);
}
// Add these specializations after the existing packed_shfl_down implementations:

/**
 * @brief Perform a packed shuffle operation synchronously across a warp.
 * @tparam T The type of the value to be shuffled.
 * @param mask[in] The mask of active threads.
 * @param f[in] The value to be shuffled.
 * @param src[in] The source lane from which to shuffle.
 * @return The result of the shuffle operation.
 */

 template<typename T>
 __attribute__((device)) static inline T packed_shfl(uint64_t mask, const T &f, int src) {
     return __shfl(f, src, 64);
 }
 template<>
 __attribute__((device)) inline bf16 packed_shfl(uint64_t mask, const bf16 &f, int src) {
     float r = __shfl(base_types::convertor<float, bf16>::convert(f), src, 64);
     return base_types::convertor<bf16, float>::convert(r);
 }

 template<>
 __attribute__((device)) inline bf16_2 packed_shfl(uint64_t mask, const bf16_2 &f, int src) {
     float2 r;
     r.x = __shfl(base_types::convertor<float, bf16>::convert(f.x), src, 64);
     r.y = __shfl(base_types::convertor<float, bf16>::convert(f.y), src, 64);
     return base_types::convertor<bf16_2, float2>::convert(r);
 }

 template<>
 __attribute__((device)) inline half packed_shfl(uint64_t mask, const half &f, int src) {
     float r = __shfl(base_types::convertor<float, half>::convert(f), src, 64);
     return base_types::convertor<half, float>::convert(r);
 }

 template<>
 __attribute__((device)) inline half_2 packed_shfl(uint64_t mask, const half_2 &f, int src) {
     float2 r;
     r.x = __shfl(base_types::convertor<float, half>::convert(f.x), src, 64);
     r.y = __shfl(base_types::convertor<float, half>::convert(f.y), src, 64);
     return base_types::convertor<half_2, float2>::convert(r);
 }

 template<>
 __attribute__((device)) inline float2 packed_shfl<float2>(uint64_t mask, const float2 &f, int src) {
     float2 r;
     r.x = __shfl(f.x, src, 64);
     r.y = __shfl(f.y, src, 64);
     return r;
 }

using bytes_4 = HIP_vector_type<float, 1>;
using bytes_8 = HIP_vector_type<float, 2>;
using bytes_16 = HIP_vector_type<float, 4>;

/* ----------  SHARED MEMORY UTILS  ---------- */

// namespace ducks {
// namespace sb {
// struct identifier {};
// }
// }

// template<typename Args...>
// struct sb {
//     using identifier = ducks::sb::identifier;
//     Args... args;
// };

// namespace ducks {
// namespace sb {
// template<typename T> concept all = requires {
//     typename T::identifier;
// } && std::is_same_v<T::identifier, identifier>;
// }
// }

// Joyously stolen from https://github.com/NVIDIA/cutlass/blob/5c447dd84f8ae0e1d48ff9a2eae26ce8c4958101/include/cute/container/alignment.hpp#L51








/**
 * @brief Dummy structure for alignment purposes. Needed for WGMMA and TMA calls.
 */
struct alignas(16) alignment_dummy { int dummy; };
/**
 * @brief Very simple allocator for dynamic shared memory. Advances pointer and tracks alignments.
 * @tparam default_alignment The default alignment this allocator will enforce. If <=0 (default -1) it will not align.
 */
template<int default_alignment=16>
struct shared_allocator {
    int *ptr;

    private:
        // Recursive template to generate N-dimensional array type
        template<typename A, size_t... dims>
        struct variadic_array;
        template<typename A, size_t first_dim, size_t... rest_dims>
        struct variadic_array<A, first_dim, rest_dims...> {
            using type = typename variadic_array<A, rest_dims...>::type[first_dim];
        };
        template<typename A>
        struct variadic_array<A> {
            using type = A;
        };
        template<typename A, size_t... dims>
        using variadic_array_t = typename variadic_array<A, dims...>::type;

        template<int alignment>
        __attribute__((device)) inline void align_ptr() {
            if constexpr (alignment > 0) {
                uint64_t p = reinterpret_cast<uint64_t>(ptr);
                if(p % alignment != 0) {
                    ptr = (int*)(p + (alignment-(p%alignment)));
                }
            }
        }

    public:
        /**
        * @brief Construct a new shared allocator using a pointer to extern shared memory.
        * @param[in] _ptr Pointer to the start of the extern shared memory.
        */
        __attribute__((device)) shared_allocator(int *_ptr): ptr(_ptr) {}
        /**
        * @brief Allocate shared memory for a single instance or N-dimensional array of type A.
        * @tparam A The type of the object to allocate.
        * @tparam dims... A list of dimensions for the N-dimensional array.
        * @return Reference to the allocated object.
        */
        template<typename A, size_t... dims>
        __attribute__((device)) inline variadic_array_t<A, dims...>& allocate() {
            // static_assert(sizeof(A) % default_alignment == 0, "Type is not aligned properly for array allocation");
            align_ptr<default_alignment>();
            using at = variadic_array_t<A, dims...>;
            at*p = reinterpret_cast<at*>(ptr);
            ptr += sizeof(at)/sizeof(int);
            return *p;
        }
        /**
        * @brief Allocate shared memory for a single instance or N-dimensional array of type A.
        * @tparam alignment An alignment to enforce for this particular object.
        * @tparam A The type of the object to allocate.
        * @tparam dims... A list of dimensions for the N-dimensional array.
        * @return Reference to the allocated object.
        */
        template<int alignment, typename A, size_t... dims>
        __attribute__((device)) inline variadic_array_t<A, dims...>& allocate() {
            // static_assert(sizeof(A) % alignment == 0, "Type is not aligned properly for array allocation");
            align_ptr<alignment>();
            using at = variadic_array_t<A, dims...>;
            at*p = reinterpret_cast<at*>(ptr);
            ptr += sizeof(at)/sizeof(int);
            return *p;
        }
};

} // namespace kittens
# 10 "/root/HipKittens//include/common/common.cuh" 2

# 1 "/root/HipKittens//include/common/base_ops.cuh" 1
/**
 * @file
 * @brief Basic operations on generic types.
 */






namespace kittens {

/**
 * @namespace base_ops
 *
 * @brief A namespace for operations on basic data types.
 */
namespace base_ops {

/* ----------  CONST OPS  ---------- */

/**
 * @brief Represents the zero constant operation.
 *
 * This operation returns the zero value of the specified type.
 *
 * @tparam T The data type for which to return the zero value.
 * @return The zero value of type T.
 */
struct zero {
    template<typename T, typename... args> __attribute__((device)) static inline constexpr T op(args... _) { return base_types::constants<T>::zero(); }
};
/**
 * @brief Represents the one constant operation.
 *
 * This operation returns the one value of the specified type.
 *
 * @tparam T The data type for which to return the one value.
 * @return The one value of type T.
 */
struct one {
    template<typename T, typename... args> __attribute__((device)) static inline constexpr T op(args... _) { return base_types::constants<T>::one(); }
};
/**
 * @brief Represents the positive infinity constant operation.
 *
 * This operation returns the positive infinity value of the specified type.
 *
 * @tparam T The data type for which to return the positive infinity value.
 * @return The positive infinity value of type T.
 */
struct pos_infty {
    template<typename T, typename... args> __attribute__((device)) static inline constexpr T op(args... _) { return base_types::constants<T>::pos_infty(); }
};
/**
 * @brief Represents the negative infinity constant operation.
 *
 * This operation returns the negative infinity value of the specified type.
 *
 * @tparam T The data type for which to return the negative infinity value.
 * @return The negative infinity value of type T.
 */
struct neg_infty {
    template<typename T, typename... args> __attribute__((device)) static inline constexpr T op(args... _) { return base_types::constants<T>::neg_infty(); }
};


/* ----------  UNARY OPS  ---------- */

/**
 * @brief Exponential function operation.
 *
 * This operation calculates the exponential of the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The exponential of the input value.
 */
struct exp {
    template<typename T> static __attribute__((device)) inline T op(const T &x) { return exp(x); }
};
template<> __attribute__((device)) inline float exp::op<float> (const float &x ) { return __expf(x); }
template<> __attribute__((device)) inline float2 exp::op<float2>(const float2 &x) { return float2{__expf(x.x), __expf(x.y)}; }
template<> __attribute__((device)) inline bf16 exp::op<bf16> (const bf16 &x ) { return hexp(x); }
template<> __attribute__((device)) inline bf16_2 exp::op<bf16_2>(const bf16_2 &x) { return h2exp(x); }
template<> __attribute__((device)) inline half exp::op<half> (const half &x ) { return hexp(x); }
template<> __attribute__((device)) inline half_2 exp::op<half_2>(const half_2 &x) { return h2exp(x); }

// /**
//  * @brief Exponential function operation, in base 2
//  *
//  * This operation calculates the exponential of the input value, in base 2.
//  *
//  * @tparam T The data type of the input and output values.
//  * @param x[in] The input value.
//  * @return The exponential of the input value.
//  */
// struct exp2 {
//     template<typename T> static __device__ inline T op(const T &x) { return exp2f(x); }
// };
// template<> __device__ inline float  exp2::op<float> (const float &x ) { return exp2f(x);                        }
// template<> __device__ inline float2 exp2::op<float2>(const float2 &x) { return float2{exp2f(x.x), exp2f(x.y)}; }
// template<> __device__ inline bf16   exp2::op<bf16>  (const bf16 &x  ) { return hexp2(x);                          }
// template<> __device__ inline bf16_2 exp2::op<bf16_2>(const bf16_2 &x) { return h2exp2(x);                         }
// template<> __device__ inline half   exp2::op<half>  (const half &x  ) { return hexp2(x);                          }
// template<> __device__ inline half_2 exp2::op<half_2>(const half_2 &x) { return h2exp2(x);                         }


/**
 * @brief Base-2 exponential operation using `__builtin_amdgcn_exp2_f32`
 *
 * Maps directly to `v_exp_f32_e32` on AMD, for highest performance.
 * Expects `x` to be in a safe numerical range (e.g., [-64, 88]).
 */
 struct exp2 {
    template <typename T>
    static __attribute__((device)) inline T op(const T &x) {
        return exp2f(x); // fallback
    }
};

// Force hardware v_exp_f32 for float
template<>
__attribute__((device)) inline float exp2::op<float>(const float &x) {
    return __builtin_amdgcn_exp2f(x); // Emits v_exp_f32_e32
}

// Force hardware v_exp_f32 for float2
template<>
__attribute__((device)) inline float2 exp2::op<float2>(const float2 &x) {
    return {
        __builtin_amdgcn_exp2f(x.x),
        __builtin_amdgcn_exp2f(x.y)
    };
}

// Delegate to low-precision approximations
template<> __attribute__((device)) inline half exp2::op<half>(const half &x) { return hexp2(x); }
template<> __attribute__((device)) inline half_2 exp2::op<half_2>(const half_2 &x) { return h2exp2(x); }
template<> __attribute__((device)) inline bf16 exp2::op<bf16>(const bf16 &x) { return hexp2(x); }
template<> __attribute__((device)) inline bf16_2 exp2::op<bf16_2>(const bf16_2 &x) { return h2exp2(x); }



/**
 * @brief Natural log function operation.
 *
 * This operation calculates the natural logarithm of the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The natural logarithm of the input value.
 */
struct log {
    template<typename T> static __attribute__((device)) inline T op(const T &x) { return log(x); }
};
template<> __attribute__((device)) inline float log::op<float> (const float &x ) { return __logf(x); }
template<> __attribute__((device)) inline float2 log::op<float2>(const float2 &x) { return float2{__logf(x.x), __logf(x.y)}; }
template<> __attribute__((device)) inline bf16 log::op<bf16> (const bf16 &x ) { return hlog(x); }
template<> __attribute__((device)) inline bf16_2 log::op<bf16_2>(const bf16_2 &x) { return h2log(x); }
template<> __attribute__((device)) inline half log::op<half> (const half &x ) { return hlog(x); }
template<> __attribute__((device)) inline half_2 log::op<half_2>(const half_2 &x) { return h2log(x); }
/**
 * @brief Logarithm base 2 operation.
 *
 * This operation calculates the logarithm base 2 of the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The logarithm base 2 of the input value.
 */
struct log2 {
    template<typename T> static __attribute__((device)) inline T op(const T &x) { return log2(x); }
};
template<> __attribute__((device)) inline float log2::op<float> (const float &x ) { return __log2f(x); }
template<> __attribute__((device)) inline float2 log2::op<float2>(const float2 &x) { return float2{__log2f(x.x), __log2f(x.y)}; }
template<> __attribute__((device)) inline bf16 log2::op<bf16> (const bf16 &x ) { return hlog2(x); }
template<> __attribute__((device)) inline bf16_2 log2::op<bf16_2>(const bf16_2 &x) { return h2log2(x); }
template<> __attribute__((device)) inline half log2::op<half> (const half &x ) { return hlog2(x); }
template<> __attribute__((device)) inline half_2 log2::op<half_2>(const half_2 &x) { return h2log2(x); }
/**
 * @brief Absolute value operation.
 *
 * This operation calculates the absolute value of the input.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The absolute value of the input.
 */
struct abs {
    template<typename T> static __attribute__((device)) inline T op(const T &x) { return abs(x); }
};
template<> __attribute__((device)) inline float abs::op<float> (const float &x ) { return fabsf(x); }
template<> __attribute__((device)) inline float2 abs::op<float2>(const float2 &x) { return float2{fabsf(x.x), fabsf(x.y)}; }
template<> __attribute__((device)) inline bf16 abs::op<bf16> (const bf16 &x ) { return __habs(x); }
template<> __attribute__((device)) inline bf16_2 abs::op<bf16_2>(const bf16_2 &x) { return __habs2(x); }
template<> __attribute__((device)) inline half abs::op<half> (const half &x ) { return __habs(x); }
template<> __attribute__((device)) inline half_2 abs::op<half_2>(const half_2 &x) { return __habs2(x); }
/**
 * @brief Rectified Linear Unit (ReLU) operation.
 *
 * This operation applies the ReLU function to the input, which is the
 * maximum of zero and the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The result of ReLU function applied to the input.
 */
struct relu {
    template<typename T> static __attribute__((device)) inline T op(const T &x) { return max(x, base_types::constants<T>::zero()); }
};
template<> __attribute__((device)) inline float relu::op<float> (const float &x ) { return max(x, 0.f); }
template<> __attribute__((device)) inline float2 relu::op<float2>(const float2 &x) { return float2{max(x.x, 0.f), max(x.y, 0.f)}; }
template<> __attribute__((device)) inline bf16 relu::op<bf16> (const bf16 &x ) { return __hmax(x, base_types::constants<bf16>::zero()); }
template<> __attribute__((device)) inline bf16_2 relu::op<bf16_2>(const bf16_2 &x) { return __hmax2(x, base_types::constants<bf16_2>::zero()); }
template<> __attribute__((device)) inline half relu::op<half> (const half &x ) { return __hmax(x, base_types::constants<half>::zero()); }
template<> __attribute__((device)) inline half_2 relu::op<half_2>(const half_2 &x) { return half_2{__hmax(x.x, base_types::constants<half>::zero()),
                                                                                     __hmax(x.y, base_types::constants<half>::zero())}; }
/**
 * @brief Copy operation.
 *
 * This operation returns the input value unchanged.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The input value.
 * @return The same value as the input.
 */
struct copy { // for non-compile-time setters.
    template<typename T> static __attribute__((device)) inline T op(const T &a) { return a; }
};


/* ----------  BINARY OPS  ---------- */

/**
 * @brief Copy2 operation.
 *
 * This operation returns the second input value unchanged.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value (ignored).
 * @param b[in] The second input value.
 * @return The same value as the second input.
 */
struct copy2 { // this turns out to be a slightly hacky op that makes some code cleaner :/
    template<typename T> static __attribute__((device)) inline T op(const T &a, const T &b) { return b; }
};
/**
 * @brief Sum operation.
 *
 * This operation calculates the sum of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The sum of the input values.
 */
struct sum {
    template<typename T> static __attribute__((device)) inline T op(const T &a, const T &b) { return a+b; }
};
template<> __attribute__((device)) inline float2 sum::op<float2>(const float2 &a, const float2 &b) { return float2{a.x+b.x, a.y+b.y}; }
template<> __attribute__((device)) inline bf16 sum::op<bf16> (const bf16 &a, const bf16 &b) { return __hadd(a, b); }
template<> __attribute__((device)) inline bf16_2 sum::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hadd2(a, b); }
template<> __attribute__((device)) inline half sum::op<half> (const half &a, const half &b) { return __hadd(a, b); }
template<> __attribute__((device)) inline half_2 sum::op<half_2>(const half_2 &a, const half_2 &b) { return __hadd2(a, b); }
/**
 * @brief Subtraction operation.
 *
 * This operation calculates the difference between two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The difference between the input values.
 */
struct sub {
    template<typename T> static __attribute__((device)) inline T op(const T &a, const T &b) { return a-b; }
};
template<> __attribute__((device)) inline float2 sub::op<float2>(const float2 &a, const float2 &b) { return float2{a.x-b.x, a.y-b.y}; }
template<> __attribute__((device)) inline bf16 sub::op<bf16> (const bf16 &a, const bf16 &b) { return __hsub(a, b); }
template<> __attribute__((device)) inline bf16_2 sub::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hsub2(a, b); }
template<> __attribute__((device)) inline half sub::op<half> (const half &a, const half &b) { return __hsub(a, b); }
template<> __attribute__((device)) inline half_2 sub::op<half_2>(const half_2 &a, const half_2 &b) { return __hsub2(a, b); }
/**
 * @brief Multiplication operation.
 *
 * This operation calculates the product of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The product of the input values.
 */
struct mul {
    template<typename T> static __attribute__((device)) inline T op(const T &a, const T &b) { return a*b; }
};
template<> __attribute__((device)) inline float2 mul::op<float2>(const float2 &a, const float2 &b) { return float2{a.x*b.x, a.y*b.y}; }
template<> __attribute__((device)) inline bf16 mul::op<bf16> (const bf16 &a, const bf16 &b) { return __hmul(a, b); }
template<> __attribute__((device)) inline bf16_2 mul::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hmul2(a, b); }
template<> __attribute__((device)) inline half mul::op<half> (const half &a, const half &b) { return __hmul(a, b); }
template<> __attribute__((device)) inline half_2 mul::op<half_2>(const half_2 &a, const half_2 &b) { return __hmul2(a, b); }
/**
 * @brief Division operation.
 *
 * This operation calculates the quotient of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The quotient of the input values.
 */
struct div {
    template<typename T> static __attribute__((device)) inline T op(const T &a, const T &b) { return a/b; }
};
template<> __attribute__((device)) inline float2 div::op<float2>(const float2 &a, const float2 &b) { return float2{a.x/b.x, a.y/b.y}; }
template<> __attribute__((device)) inline bf16 div::op<bf16> (const bf16 &a, const bf16 &b) { return __hdiv(a, b); }
template<> __attribute__((device)) inline bf16_2 div::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __h2div(a, b); } // this op is a special snowflake
template<> __attribute__((device)) inline half div::op<half> (const half &a, const half &b) { return __hdiv(a, b); }
template<> __attribute__((device)) inline half_2 div::op<half_2>(const half_2 &a, const half_2 &b) { return __h2div(a, b); }
/**
 * @brief Maximum operation.
 *
 * This operation calculates the maximum of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The maximum of the input values.
 */
 struct max {
    template<typename T> static __attribute__((device)) inline T op(const T &a, const T &b) { return ::max(a, b); }
};
template<> __attribute__((device)) inline float2 max::op<float2>(const float2 &a, const float2 &b) { return float2{::max(a.x, b.x), ::max(a.y, b.y)}; }
template<> __attribute__((device)) inline bf16 max::op<bf16> (const bf16 &a, const bf16 &b) { return __hmax(a, b); }
template<> __attribute__((device)) inline bf16_2 max::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hmax2(a, b); }
template<> __attribute__((device)) inline half max::op<half> (const half &a, const half &b) { return __hmax(a, b); }
template<> __attribute__((device)) inline half_2 max::op<half_2>(const half_2 &a, const half_2 &b) { return half_2{__hmax(a.x, b.x), __hmax(a.y, b.y)}; }
/**
 * @brief Minimum operation.
 *
 * This operation calculates the minimum of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The minimum of the input values.
 */
struct min {
    template<typename T> static __attribute__((device)) inline T op(const T &a, const T &b) { return ::min(a, b); }
};
template<> __attribute__((device)) inline float2 min::op<float2>(const float2 &a, const float2 &b) { return float2{::min(a.x, b.x), ::min(a.y, b.y)}; }
template<> __attribute__((device)) inline bf16 min::op<bf16> (const bf16 &a, const bf16 &b) { return __hmin(a, b); }
template<> __attribute__((device)) inline bf16_2 min::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hmin2(a, b); }
template<> __attribute__((device)) inline half min::op<half> (const half &a, const half &b) { return __hmin(a, b); }
template<> __attribute__((device)) inline half_2 min::op<half_2>(const half_2 &a, const half_2 &b) { return half_2{__hmin(a.x, b.x), __hmin(a.y, b.y)}; }


/* ----------  TERNARY OPS  ---------- */

/**
 * @brief Fused multiply-add operation A * B + C.
 *
 * This operation performs a fused multiply-add, computing (A * B) + C with only one rounding.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @param c[in] The third input value to be added.
 * @return The result of the fused multiply-add operation.
 */
struct fma_AxBtC {
    template<typename T> static __attribute__((device)) inline T op(const T &a, const T &b, const T &c) {
        return sum::op<T>(mul::op<T>(a, b), c);
    }
};
/**
 * @brief Fused multiply-add operation A * C + B.
 *
 * This operation performs a fused multiply-add, computing (A * C) + B with only one rounding.
 * This is particularly useful for attention mechanisms in neural networks.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The third input value to be added.
 * @param c[in] The second input value.
 * @return The result of the fused multiply-add operation.
 */
struct fma_AxCtB { // this is the one needed for attention
    template<typename T> static __attribute__((device)) inline T op(const T &a, const T &b, const T &c) {
        return sum::op<T>(mul::op<T>(a, c), b);
    }
};

} // namespace base_ops

} // namespace kittens
# 11 "/root/HipKittens//include/common/common.cuh" 2
# 9 "/root/HipKittens//include/kittens.cuh" 2
# 1 "/root/HipKittens//include/types/types.cuh" 1
/**
 * @file
 * @brief An aggregate header file for all the register and shared types defined by ThunderKittens.
 */



# 1 "/root/HipKittens//include/types/register/register.cuh" 1
/**
 * @file
 * @brief An aggregate header file for all the register types defined by ThunderKittens.
 */



# 1 "/root/HipKittens//include/types/register/rv_layout.cuh" 1
/**
 * @file
 * @brief Layouts and their manipulations for register tiles.
 */





namespace kittens {
namespace ducks {
/**
 * @namespace rv_layout
 * 
 * @brief A namespace for template metaprogramming with register vector layouts.
 */
namespace rv_layout {

/**
 * @brief A dummy type used to identify an aligned (32x replicated) layout.
 */
struct align { constexpr static int inner_dim = 2; };
/**
 * @brief A dummy type used to identify an orthogonal (2x replicated) layout.
 */
struct ortho { constexpr static int inner_dim = 1; };
/**
 * @brief A dummy type used to identify an unreplicated layout, for better coalesced loads and vector operations like layernorm.
 */
struct naive { constexpr static int inner_dim = 1; };

/**
 * @brief A concept to check if a type is a register tile layout.
 */
template<typename T>
concept all = std::is_same_v<T, align> || std::is_same_v<T, ortho> || std::is_same_v<T, naive>;

} // namespace rv_layout
} // namespace ducks
} // namespace kittens
# 9 "/root/HipKittens//include/types/register/register.cuh" 2
# 1 "/root/HipKittens//include/types/register/rt_base.cuh" 1
/**
 * @file
 * @brief The basic 16x16 register tile on which larger register tiles are built.
 */






# 1 "/root/HipKittens//include/types/register/rt_layout.cuh" 1
/**
 * @file
 * @brief Layouts and their manipulations for register tiles.
 */





namespace kittens {
namespace ducks {
/**
 * @namespace rt_layout
 * 
 * @brief A namespace for template metaprogramming with register tile layouts.
 */
namespace rt_layout {

/**
 * @brief A dummy type used to identify a row-major layout for a register tile.
 */
struct row {}; // for most matrices
/**
 * @brief A dummy type used to identify a col-major layout for a register tile.
 */
struct col {}; // for the B-matrix of MMA ops.

/**
 * @brief A concept to check if a type is a register tile layout.
 */
template<typename T>
concept all = std::is_same_v<T, row> || std::is_same_v<T, col>;

/**
 * @brief A struct to generate a transposed layout.
 */
template<all L> struct transpose { using type = col; };
template<> struct transpose<col> { using type = row; };

} // namespace rt_layout
} // namespace ducks
} // namespace kittens
# 12 "/root/HipKittens//include/types/register/rt_base.cuh" 2


namespace kittens {

/* ----------  BASE 16x16 SUBTILE STRUCT  ---------- */

namespace ducks {
/**
 * @namespace rt_base
 * 
 * @brief The namespace where concepts and abstract types for register base (16x16) tiles live.
 */
namespace rt_base {
/**
 * @brief A dummy type used to identify register base tiles.
 * 
 * For a type to quack like an rt_base, it should define its identifier as ducks::rt_base::identifier.
 * If a type quacks like ducks::rt_base::identifier, it will be treated as an rt_base by compiler checks.
 */
struct identifier {};
}
} // namespace ducks

/**
 * @brief Basic tile structure for computation in registers.
 *
 * @tparam T2 The packed data type used for the matrix elements.
 * @tparam _layout The layout of the base tile, either row-major or column-major.
 *
 * This type is a primarily utility for building larger inline templates
 * out of PTX primitives and managing layouts.
 * 
 * In general, you probably want a row-major tile, unless you specifically want to call mma
 */
template<typename _T, ducks::rt_layout::all _layout> struct rt_base {
    using identifier = ducks::rt_base::identifier; ///< Type identifier for the rt_base structure.
    using layout = _layout; ///< Layout of the matrix tile.
    static_assert(kittens::ducks::base_types::T1<_T>); // confirm it's a supported type
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = T2; ///< Data type of the matrix elements

    static_assert(
        std::is_same_v<dtype, bf16_2> || std::is_same_v<dtype, float2> || std::is_same_v<dtype, half_2>,
        "rt_base was provided an unsupported type."
    );

    static constexpr int tile_size_row = kittens::TILE_ROW_DIM<T>;
    static constexpr int tile_size_col = kittens::TILE_COL_DIM<T>;
    static constexpr int rows = tile_size_row; ///< Number of rows.
    static constexpr int cols = tile_size_col; ///< Number of cols.
    static constexpr int num_elements = rows*cols;
    static constexpr int elements_per_thread = num_elements / kittens::WARP_THREADS;

    static constexpr int packed_per_thread = (elements_per_thread / base_types::packing<dtype>::num()) ;
    static constexpr int registers_per_thread = packed_per_thread * sizeof(dtype) / 4; // registers are 32-bit words

    using row_vec_layout = std::conditional_t<std::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::align, ducks::rv_layout::ortho>; // for holding column reductions
    using col_vec_layout = std::conditional_t<std::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::ortho, ducks::rv_layout::align>; // for holding row reductions

    dtype data[packed_per_thread]; ///< The actual storage for the base tile
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace rt_base {
/**
* @brief Concept for all register base tiles.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as rt_base::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::rt::identifier
} // namespace rt
} // namespace ducks

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_fl = rt_base<float, L>;
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_bf = rt_base<bf16, L>;
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_hf = rt_base<half, L>;
}
# 10 "/root/HipKittens//include/types/register/register.cuh" 2
# 1 "/root/HipKittens//include/types/register/rv.cuh" 1
/**
 * @file
 * @brief Register vectors for computations on axes.
 */
# 14 "/root/HipKittens//include/types/register/rv.cuh"
namespace kittens {

/* ----------  MAIN VECTOR STRUCT  ---------- */

// helper struct for type inference
namespace ducks {
/**
 * @namespace rt
 * 
 * @brief The namespace where concepts and abstract types for register vectors live.
 */
namespace rv {
/**
 * @brief A dummy type used to identify register vectors.
 * 
 * For a type to quack like an rv, it should define its identifier as ducks::rv::identifier.
 * If a type quacks like ducks::rv::identifier, it will be treated as an rv by compiler checks.
 */
struct identifier {};
}
}
/**
 * @brief Register vector structure.
 *
 * @tparam _T The packed data type used for the vector elements.
 * @tparam _outer_dim The size of the tile, in units of TILE_DIM (16).
 * @tparam _inner_dim This controls the layout of the tile in terms of which axis it maps on the register tile layout.
 *
 * Register vectors are used to accumulate and map values across tiles. You can do computation
 * on them directly if you want, but they're not designed to be maximally efficient vectors
 * as they have substantial duplication and strange layouts to help them work efficiently with
 * the register layouts used by the tensor cores. ThunderKittens wants you working with tiles
 * where possible!
 */
template<typename _T, size_t _length, ducks::rv_layout::all _layout=ducks::rv_layout::naive>
struct rv {
    using identifier = ducks::rv::identifier; ///< Type identifier for the rv structure.
    static_assert(kittens::ducks::base_types::T1<_T>); // confirm it's a supported type
    using layout = _layout;
    static constexpr bool is_naive = std::is_same_v<layout, ducks::rv_layout::naive>;
    static constexpr bool is_ortho = std::is_same_v<layout, ducks::rv_layout::ortho>;
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = std::conditional_t<is_naive || is_ortho, T, T2>; ///< Data type of the matrix elements

    static constexpr int length = _length; ///< Length in elements.
    static_assert(length % kittens::TILE_ROW_DIM<T> == 0, "Length must be divisible by the tile dimension");
    static constexpr int tiles = _length / kittens::TILE_ROW_DIM<T>; ///< Length in subtiles, aliased for consistency with sv type
    static constexpr int inner_dim = layout::inner_dim; ///< Internal layout within a subtile. Either 1 or 2.
    static constexpr int outer_dim = is_naive ? (tiles+3)/4 : tiles; ///< Outer dim (also length in tiles)

    dtype data[outer_dim][inner_dim]; ///< The actual register vector data.

    __attribute__((device)) inline dtype* operator[](size_t idx) { return &data[idx][0]; } ///< A wrapper for indexing into vector data.
    __attribute__((device)) inline const dtype* operator[](size_t idx) const { return &data[idx][0]; } ///< A wrapper for indexing into vector data.
    __attribute__((device)) inline dtype& operator[](int2 outin) { return data[outin.x][outin.y]; } ///< A wrapper for indexing into vector data.
    __attribute__((device)) inline const dtype& operator[](int2 outin) const { return data[outin.x][outin.y]; } ///< A wrapper for indexing into vector data.
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace rv {
/**
* @brief Concept for all register vectors.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as rv::identifier.
*/
template<typename T>
concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::rv::identifier.

template<typename T> concept naive_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::naive>;
template<typename T> concept align_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::align>;
template<typename T> concept ortho_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::ortho>;
template<typename T> concept tile_layout = align_layout<T> || ortho_layout<T>; // vector layouts for interacting with tiles.

} // namespace rv
} // namespace ducks

template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_fl = rv<float, _l, layout>;
template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_bf = rv<bf16, _l, layout>;
template<int _l, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_hf = rv<half, _l, layout>;

} // namespace kittens
# 11 "/root/HipKittens//include/types/register/register.cuh" 2
# 1 "/root/HipKittens//include/types/register/rt.cuh" 1
/**
 * @file
 * @brief The main ThunderKittens register tile struct, where most computation happens.
 */
# 17 "/root/HipKittens//include/types/register/rt.cuh"
namespace kittens {

/* ----------  MAIN TILE STRUCT  ---------- */

// helper struct for type inference
namespace ducks {
/**
 * @namespace rt
 * 
 * @brief The namespace where concepts and abstract types for register tiles live.
 */
namespace rt {
/**
 * @brief A dummy type used to identify register tiles.
 * 
 * For a type to quack like an rt, it should define its identifier as ducks::rt::identifier.
 * If a type quacks like ducks::rt::identifier, it will be treated as an rt by compiler checks.
 */
struct identifier {};
} // namespace rt
} // namespace ducks

/**
 * @brief Main tile structure for manipulating data in registers.
 *
 * @tparam T2 The packed data type used for the matrix elements.
 * @tparam _height The height of the tile in terms of the number of subtiles.
 * @tparam _width The width of the tile in terms of the number of subtiles.
 * @tparam _layout The layout of the internal base tiles, either row-major or column-major.
 *
 * This structure is designed to handle matrix tiles in a flexible manner, allowing
 * for operations on tiles that are composed of smaller subtiles. It supports both
 * row-major and column-major layouts and includes helper structs for type inference
 * in vector maps.
 * 
 * In general, you probably want a row-major tile, unless you specifically want to call mma
 */
template<typename _T, int _rows, int _cols, ducks::rt_layout::all _layout=ducks::rt_layout::row>
struct rt {
    using identifier = ducks::rt::identifier; ///< Type identifier for the rt structure.
    using layout = _layout; ///< Layout of the matrix tile.
    static_assert(kittens::ducks::base_types::T1<_T>); // confirm it's a supported type
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = T2; ///< Data type of the matrix elements

    static constexpr int rows = _rows; ///< Total number of rows.
    static_assert(rows % rt_base<T, layout>::tile_size_row == 0, "Rows must be divisible by the tile size");
    static constexpr int cols = _cols; ///< Total number of columns.
    static_assert(cols % rt_base<T, layout>::tile_size_col == 0, "Columns must be divisible by the tile size");
    static constexpr int height = rows / rt_base<T, layout>::tile_size_row; ///< Height in subtiles.
    static constexpr int width = cols / rt_base<T, layout>::tile_size_col; ///< Width in subtiles.
    static constexpr int tile_size_row = rt_base<T, layout>::tile_size_row; ///< Size of the base tile.
    static constexpr int tile_size_col = rt_base<T, layout>::tile_size_col; ///< Size of the base tile.
    static constexpr int num_elements = rt_base<T, layout>::num_elements * width * height; ///< Total number of elements.
    static constexpr int elements_per_thread = rt_base<T, layout>::elements_per_thread * width * height; ///< Elements handled per thread.
    static constexpr int packed_per_thread = rt_base<T, layout>::packed_per_thread * width * height; ///< Packed elements per thread.
    static constexpr int packed_per_tile = rt_base<T, layout>::packed_per_thread; ///< Packed elements per tile.

    rt_base<T, layout> tiles[height][width]; ///< The actual storage for the matrix tile, organized in subtiles.

    using row_vec = rv<T, cols, typename rt_base<T, layout>::row_vec_layout>; ///< A type representing a column vector for this tile.
    using col_vec = rv<T, rows, typename rt_base<T, layout>::col_vec_layout>; ///< A type representing a column vector for this tile.
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace rt {
/**
* @brief Concept for all register tiles.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as rt::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::rt::identifier
/**
* @brief Concept for register tiles with row layout.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a register tile.
* - T has an internal type layout that is ducks::rt_layout::row.
*/
template<typename T>
concept row_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::row>;
/**
* @brief Concept for register tiles with col layout.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a register tile.
* - T has an internal type layout that is ducks::rt_layout::col.
*/
template<typename T>
concept col_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::col>;
} // namespace rt
} // namespace ducks


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// layout and type wrappers

template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl = rt<float, _r, _c, layout>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf = rt<bf16, _r, _c, layout>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_hf = rt<half, _r, _c, layout>;

} // namespace kittens
# 12 "/root/HipKittens//include/types/register/register.cuh" 2
# 9 "/root/HipKittens//include/types/types.cuh" 2
# 1 "/root/HipKittens//include/types/shared/shared.cuh" 1
/**
 * @file
 * @brief An aggregate header file for all the shared types defined by ThunderKittens.
 */



# 1 "/root/HipKittens//include/types/shared/sv.cuh" 1
/**
 * @file
 * @brief The ThunderKittens shared vector struct.
 */
# 13 "/root/HipKittens//include/types/shared/sv.cuh"
namespace kittens {

/* ----------  MAIN VECTOR STRUCT  ---------- */

namespace ducks {
/**
 * @namespace sv
 * 
 * @brief The namespace where concepts and abstract types for shared vectors live.
 */
namespace sv {
/**
 * @brief A dummy type used to identify shared vectors.
 * 
 * For a type to quack like an sv, it should define its identifier as ducks::sv::identifier.
 * If a type quacks like ducks::sv::identifier, it will be treated as an sv by compiler checks.
 */
struct identifier {};
}
}

/**
 * @brief Shared vector structure.
 *
 * @tparam _T The packed data type used for the vector elements.
 * @tparam _tiles The size of the tile, in units of TILE_ROW_DIM (16 for fp16, bf16, fp32).
 *
 * Shared vectors are used to accumulate and map values across shared tiles.
 * Unlike every other structure present in ThunderKittens, these have a simple
 * uniform layout which is just an array in memory. EZ!
 */
template<typename _T, size_t _length>
struct alignas(16) sv {
    using identifier = ducks::sv::identifier;
    using T = base_types::packing<_T>::unpacked_type;
    using T2 = base_types::packing<_T>::packed_type;
    using dtype = T; ///< Data type of the elements in the tile.

    static constexpr int length = _length; ///< Length in elements.
    static_assert(length % TILE_ROW_DIM<T> == 0, "Length must be divisible by the tile dimension");
    static constexpr int tiles = length / TILE_ROW_DIM<T>; ///< Length in subtiles.'

    static constexpr int num_alloc_elements = length;

    dtype data[num_alloc_elements]; ///< The actual shared vector data.

    __attribute__((device)) static inline T* idx(T *ptr, int idx) { // useful for computations in shared address space, as silly as it sounds.
        return ptr[idx];
    }

    __attribute__((device)) inline dtype& operator[](size_t idx) { return data[idx]; }
    __attribute__((device)) inline const dtype& operator[](size_t idx) const { return data[idx]; }

    template<size_t sub_length> using subvec = sv<dtype, sub_length>; ///< A subvector which allows warpgroups and blocks to work cooperatively.
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace sv {
/**
* @brief Concept for all shared vectors.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as sv::identifier.
*/
template<typename T>
concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::sv::identifier

} // namespace sv
} // namespace ducks


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// vector types
template<size_t _length> using sv_bf = sv<bf16, _length>;
template<size_t _length> using sv_hf = sv<half, _length>;
template<size_t _length> using sv_fl = sv<float, _length>;

} // namespace kittens
# 9 "/root/HipKittens//include/types/shared/shared.cuh" 2
# 1 "/root/HipKittens//include/types/shared/st.cuh" 1
/**
 * @file
 * @brief The ThunderKittens shared tile struct.
 */






# 1 "/root/HipKittens//include/types/shared/st_layout.cuh" 1
/**
 * @file
 * @brief Layout concepts for shared memory tiles.
 */





namespace kittens {
namespace ducks {
/**
 * @namespace st_layout
 * 
 * @brief A namespace for template metaprogramming with shared memory tile layouts.
 */
namespace st_layout {

/**
 * @brief A dummy type used to identify a row-major layout for a shared memory tile.
 */
struct row {}; // for most matrices
/**
 * @brief A dummy type used to identify a col-major layout for a shared memory tile.
 */
struct col {}; // for the B-matrix of MMA ops.

/**
 * @brief A concept to check if a type is a shared memory tile layout.
 */
template<typename T>
concept all = std::is_same_v<T, row> || std::is_same_v<T, col>;

} // namespace st_layout
} // namespace ducks
} // namespace kittens
# 12 "/root/HipKittens//include/types/shared/st.cuh" 2

/* ----------  MAIN TILE STRUCT  ---------- */

// these are helper structs for type inference
namespace kittens {
    namespace ducks {
    /**
     * @namespace rt
     * 
     * @brief The namespace where concepts and abstract types for shared tiles live.
     */
    namespace st {
    /**
     * @brief A dummy type used to identify shared tiles.
     * 
     * For a type to quack like an st, it should define its identifier as ducks::st::identifier.
     * If a type quacks like ducks::st::identifier, it will be treated as an st by compiler checks.
     * This is particularly useful for subtiles.
     */
    struct identifier {};
    }
    } // namespace ducks

    // Forward declaration of subtile
    template<
        typename ST,
        int _subtile_height,
        int _subtile_width
    >
    struct st_subtile;

    /**
     * @brief Shared memory tile structure for various data types and layouts.
     *
     * @tparam T The data type of the elements in the tile. Not packed!
     * @tparam _rows The height of the tile.
     * @tparam _cols The width of the tile.
     */
    template<typename _T, int _rows, int _cols>
    struct alignas(16) st {
        using identifier = ducks::st::identifier; ///< Type identifier for shared memory tile.
        using T = base_types::packing<_T>::unpacked_type;
        using T2 = base_types::packing<_T>::packed_type;
        using dtype = T; ///< Data type of the elements in the tile.

        // define underlying data as same as that projected, to make clear that this is *not* a subtile.
        static constexpr int underlying_rows = _rows;
        static constexpr int underlying_cols = _cols;
        static constexpr int underlying_height = _rows / kittens::TILE_ROW_DIM<T>;
        static constexpr int underlying_width = _cols / kittens::TILE_COL_DIM<T>;
        static constexpr int underlying_num_elements = underlying_rows * underlying_cols;

        static constexpr int rows = _rows; ///< Total number of rows in the tile.
        static_assert(rows % kittens::TILE_ROW_DIM<T> == 0, "Rows must be divisible by the tile dimension");
        static constexpr int cols = _cols; ///< Total number of cols in the tile.
        static_assert(cols % kittens::TILE_COL_DIM<T> == 0, "Cols must be divisible by the tile dimension");
        static constexpr int height = _rows / kittens::TILE_ROW_DIM<T>; ///< Height of the tile in terms of 16-element subtiles.
        static constexpr int width = _cols / kittens::TILE_COL_DIM<T>; ///< Width of the tile in terms of 16-element subtiles.
        static constexpr int num_elements = rows * cols; ///< Total number of elements in the tile.

        static_assert(base_types::packing<dtype>::num() == 1); // must be a 1-packed type (e.g. float, bf16, etc)

        static constexpr int swizzle_bytes = (
            sizeof(dtype) == 1 ? (
                underlying_width%4 == 0 ? 128 :
                underlying_width%2 == 0 ? 64 : 32
            ) :
            sizeof(dtype) == 2 ? (
                underlying_width%4 == 0 ? 128 :
                underlying_width%2 == 0 ? 64 : 32
            ) :
            sizeof(dtype) == 4 ? (
                underlying_width%2 == 0 ? 128 : 64
            ) : -1
        );
        static constexpr int swizzle_repeat = swizzle_bytes << 4;
        static constexpr int subtile_cols = swizzle_bytes / sizeof(T);

        dtype data[rows*cols]; ///< Raw data storage for the tile.

        __attribute__((device)) static inline T* idx(T *ptr, int2 coord) { // naive row-major coord default
            int r = coord.x, c = coord.y; // alias
            const int outer_idx = c/subtile_cols;
            const uint64_t addr = (uint64_t)(&ptr[outer_idx*rows*subtile_cols + r*subtile_cols + c%subtile_cols]);
            const int swizzle = ((addr % swizzle_repeat) >> 7) << 3;

            return (T*)(addr ^ swizzle);
        }
        __attribute__((device)) static inline uint32_t idx(uint32_t ptr, int2 coord) {
            int r = coord.x, c = coord.y; // alias
            const int outer_idx = c/subtile_cols;
            const uint32_t addr = ptr + sizeof(T)*(outer_idx*rows*subtile_cols + r*subtile_cols + c%subtile_cols);
            const int swizzle = ((addr % swizzle_repeat) >> 7) << 3;

            return (addr ^ swizzle);
        }
        /**
         * @brief Access a shared tile element using a row and column, as if the tile were row-major.
         *
         * This is the preferred way to access memory within a shared tile, which abstracts
         * indexing calculations for swizzled layouts.
         */
        __attribute__((device)) inline dtype& operator[](const int2 &rowcol) {
            return *idx(data, rowcol);
        }
        __attribute__((device)) inline const dtype& operator[](const int2 &rowcol) const {
            return *(const dtype*)idx((dtype*)data, rowcol);
        }
        __attribute__((device)) inline dtype& operator[](int idx) {
            return data[idx];
        }
        __attribute__((device)) inline const dtype& operator[](int idx) const {
            return data[idx];
        }

        // vector types
        using col_vec = sv<dtype, rows>; ///< Column vector type for this tile
        using row_vec = sv<dtype, cols>; ///< Row vector type for this tile
        template<int subtile_rows, int subtile_cols> using subtile = st_subtile<
            st<T, rows, cols>, subtile_rows, subtile_cols
        >; ///< A templated subtile type wrapper for this tile.
    };


    /**
     * @brief A reference into a chunk of shared tile memory.
     *
     * The st_subtile is a drop-in replacement for an st which internally
     * references the appropriate memory while performing minimal address
     * calculations. You should never create this directly, but instead
     * have subtile_inplace return it for you instead. (`auto` is nice.)
     *
     * You can generally just pretend this is an st. But not for wgmma's.
     */
    template<
        typename _ST,
        int _subtile_rows,
        int _subtile_cols
    >
    struct st_subtile {
        using identifier = ducks::st::identifier; // i quack like an st, gcc will never know the difference
        using ST = _ST;
        using T = ST::T;
        using T2 = ST::T2;
        using dtype = T; ///< Data type of the elements in the tile.

        static constexpr int underlying_rows = ST::underlying_rows;
        static_assert(underlying_rows % kittens::TILE_ROW_DIM<T> == 0, "Underlying rows must be divisible by the tile dimension");
        static constexpr int underlying_cols = ST::underlying_cols;
        static_assert(underlying_cols % kittens::TILE_COL_DIM<T> == 0, "Underlying cols must be divisible by the tile dimension");
        static constexpr int underlying_height = ST::underlying_height;
        static constexpr int underlying_width = ST::underlying_width;
        static constexpr int underlying_num_elements = ST::underlying_num_elements;

        static constexpr int rows = _subtile_rows;
        static_assert(rows % kittens::TILE_ROW_DIM<T> == 0, "Rows must be divisible by the tile dimension");
        static constexpr int cols = _subtile_cols;
        static_assert(cols % kittens::TILE_COL_DIM<T> == 0, "Cols must be divisible by the tile dimension");
        static constexpr int height = rows / kittens::TILE_ROW_DIM<T>;
        static constexpr int width = cols / kittens::TILE_COL_DIM<T>;
        static constexpr int num_elements = rows * cols;

        static constexpr int swizzle_bytes = ST::swizzle_bytes;
        static constexpr int swizzle_repeat = ST::swizzle_repeat;
        static constexpr int subtile_cols = ST::subtile_cols;
        dtype *data;
        int row_offset, col_offset;

        __attribute__((device)) st_subtile(ST &src, int2 rowcol) {
            data = &src.data[0];
            row_offset = rowcol.x * rows;
            col_offset = rowcol.y * cols;
        }

        __attribute__((device)) inline T* idx(T *ptr, const int2 coord) { // naive row-major coord default
            int r = coord.x+row_offset, c = coord.y+col_offset; // alias
            const int outer_idx = c/subtile_cols;
            const uint64_t addr = (uint64_t)(&ptr[outer_idx*underlying_rows*subtile_cols + r*subtile_cols + c%subtile_cols]);
            const int swizzle = ((addr % swizzle_repeat) >> 7) << 3;

            return (T*)(addr ^ swizzle);
        }
        __attribute__((device)) inline const T* idx(const T *ptr, const int2 coord) const { // const version
            int r = coord.x+row_offset, c = coord.y+col_offset; // alias
            const int outer_idx = c/subtile_cols;
            const uint64_t addr = (uint64_t)(&ptr[outer_idx*underlying_rows*subtile_cols + r*subtile_cols + c%subtile_cols]);
            const int swizzle = ((addr % swizzle_repeat) >> 7) << 3;

            return (const T*)(addr ^ swizzle);
        }
        __attribute__((device)) inline uint32_t idx(uint32_t ptr, const int2 coord) const { // naive row-major coord default
            int r = coord.x+row_offset, c = coord.y+col_offset; // alias
            const int outer_idx = c/subtile_cols;
            const uint32_t addr = ptr + sizeof(T)*(outer_idx*underlying_rows*subtile_cols + r*subtile_cols + c%subtile_cols);
            const int swizzle = ((addr % swizzle_repeat) >> 7) << 3;

            return (addr ^ swizzle);
        }
        /**
         * @brief Access a shared tile element using a row and column, as if the tile were row-major.
         *
         * This is the preferred way to access memory within a shared tile, which abstracts
         * indexing calculations for swizzled layouts.
         */
        __attribute__((device)) inline dtype& operator[](const int2 &rowcol) {
            return *idx(data, rowcol);
        }
        __attribute__((device)) inline const dtype& operator[](const int2 &rowcol) const {
            return *(const dtype*)idx((dtype*)data, rowcol);
        }

        // single-coord operator[] is left undefined as it would likely be an improper use of st_subtile type.
        // can of course be end-run by just accessing .data directly.

        // vector types
        using col_vec = sv<dtype, rows>;
        using row_vec = sv<dtype, cols>;
    };

    /* ----------  CONCEPTS  ---------- */

    namespace ducks {
    namespace st {

    /**
    * @brief Concept for all shared tiles.
    * @tparam T The type to check against the concept requirements.
    *
    * Requires:
    * - T has a nested type identifier that is the same as st::identifier.
    */
    template<typename T> concept all = requires {
        typename T::identifier; // Checks if T::identifier exists
    } && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::st::identifier

    } // namespace st
    } // namespace ducks


    /* ----------  WRAPPERS FOR PRETTINESS  ---------- */

    template<int _height, int _width> using st_bf = st<bf16, _height, _width>;
    template<int _height, int _width> using st_hf = st<half, _height, _width>;
    template<int _height, int _width> using st_fl = st<float, _height, _width>;
}
# 10 "/root/HipKittens//include/types/shared/shared.cuh" 2
# 10 "/root/HipKittens//include/types/types.cuh" 2
# 1 "/root/HipKittens//include/types/global/global.cuh" 1
/**
 * @file
 * @brief An aggregate header file for all the global types defined by ThunderKittens.
 */



# 1 "/root/HipKittens//include/types/global/util.cuh" 1





namespace kittens {
namespace ducks {
namespace gl {

template<int d> concept cdim = (d > 0); // represents a compile-time dimension
template<int d> concept rdim = (d == -1); // represents a runtime dimension
template<int _v> struct compiled_dim {
    static_assert(cdim<_v>, "Invalid compile-time dimension value");
    static constexpr size_t v = _v;
    __attribute__((host)) __attribute__((device)) inline compiled_dim(const std::nullptr_t &_) {}
    __attribute__((host)) __attribute__((device)) inline constexpr operator size_t() const { return v; }
};
struct runtime_dim {
    size_t v;
    __attribute__((host)) __attribute__((device)) inline runtime_dim(const size_t &_v) : v(_v) {}
    __attribute__((host)) __attribute__((device)) inline operator size_t() const { return v; }
};
template<int d> using make_dim_t = std::conditional_t<rdim<d>, runtime_dim, compiled_dim<d>>;
template<int d> using make_arg_t = std::conditional_t<rdim<d>, size_t, std::nullptr_t>; // we pass runtime dims as size_t, comptime dims as nullptr_t
}
}

namespace detail {
template<typename T> concept tile = ducks::st::all<T> || ducks::rt::all<T>;
template<typename T> concept vec = ducks::sv::all<T> || ducks::rv::all<T>;
}

namespace ducks {
namespace coord {
struct identifier {};
}
}
template<typename _T=ducks::default_type> struct coord { // essentially a named int4 for tensor coordinates.
    using identifier = ducks::coord::identifier;
    using BASE = _T; // in units of what type?
    // static_assert(std::is_same_v<BASE, ducks::default_type> || detail::tile<BASE> || detail::vec<BASE>); // ensure BASE is a valid type
    int b, d, r, c;
    __attribute__((device)) inline coord(int _b, int _d, int _r, int _c) : b(_b), d(_d), r(_r), c(_c) {}
    __attribute__((device)) inline coord( int _d, int _r, int _c) : b( 0), d(_d), r(_r), c(_c) {}
    __attribute__((device)) inline coord( int _r, int _c) : b( 0), d( 0), r(_r), c(_c) {}
    __attribute__((device)) inline coord( int _c) : b( 0), d( 0), r( 0), c(_c) {}
    __attribute__((device)) inline coord( ) : b( 0), d( 0), r( 0), c( 0) {}
    template<typename U> __attribute__((device)) inline coord(const coord<U> &other) : b(other.b), d(other.d), r(other.r), c(other.c) {}
    __attribute__((device)) inline coord(const int4 &other) : b(other.x), d(other.y), r(other.z), c(other.w) {}
    __attribute__((device)) inline operator int4() const { return int4(b, d, r, c); }
    template<int row_axis, int col_axis> __attribute__((device)) inline coord<ducks::default_type> unit_coord() const {
        if constexpr (detail::tile<BASE>) {
            static_assert(row_axis != col_axis, "row and column axes must be different");
            static_assert(row_axis >= 0 && row_axis <= 3, "row axis must be between 0 and 3");
            static_assert(col_axis >= 0 && col_axis <= 3, "column axis must be between 0 and 3");
            static_assert(col_axis == 3, "for now, column axis must be 3");
            return coord<ducks::default_type>(
                row_axis == 0 ? b*BASE::rows : b,
                row_axis == 1 ? d*BASE::rows : d,
                row_axis == 2 ? r*BASE::rows : r,
                c*BASE::cols
            );
        }
        else if constexpr (detail::vec<BASE>) {
            static_assert(row_axis == -1, "row axis must be be -1 for a vector coordinate to be converted to a unit coordinate");
            static_assert(col_axis >= 0 && col_axis <= 3, "column axis must be between 0 and 3");
            static_assert(col_axis == 3, "for now, column axis must be 3");
            return coord<ducks::default_type>(b, d, r, c*BASE::length);
        }
        else {
            return coord<ducks::default_type>(*this);
        }
    }
    template<int axis> __attribute__((device)) inline int dim() const {
        static_assert(axis >= 0 && axis <= 3, "axis must be between 0 and 3");
        if constexpr (axis == 0) { return b; }
        else if constexpr (axis == 1) { return d; }
        else if constexpr (axis == 2) { return r; }
        else { return c; }
    }
};
namespace ducks {
namespace coord {
/**
* @brief Concept for all coordinate types.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as ducks::coord::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::coord::identifier
template<typename T> concept tile = all<T> && (std::is_same_v<typename T::BASE, ducks::default_type> || detail::tile<typename T::BASE>);
template<typename T> concept vec = all<T> && (std::is_same_v<typename T::BASE, ducks::default_type> || detail::vec<typename T::BASE>);
}
}
}
# 9 "/root/HipKittens//include/types/global/global.cuh" 2
# 1 "/root/HipKittens//include/types/global/gl.cuh" 1
/**
 * @file
 * @brief Templated layouts for global memory.
 */







namespace kittens {

/* ----------   Associative dictionary for global layouts  ---------- */

namespace detail {
template<typename... Args>
struct descriptor_dict {
    __attribute__((host)) descriptor_dict() {}
    template<typename T> __attribute__((host)) descriptor_dict(T _, int b, int d, int r, int c) {}
    __attribute__((host)) __attribute__((device)) descriptor_dict(const descriptor_dict &other) {}
};
}

/* ----------  Global layout descriptor  ---------- */

namespace ducks {
namespace gl {
struct identifier {};
}
}

template<typename _T, int b, int d, int r, int c, typename... TMA_Types>
struct gl {
    using identifier = ducks::gl::identifier;

    using T = base_types::packing<_T>::unpacked_type;
    using T2 = base_types::packing<_T>::packed_type;
    using dtype = T;

    T* raw_ptr;

    static constexpr int __b__ = b, __d__ = d, __r__ = r, __c__ = c; // Not to be touched by the user.

    ducks::gl::make_dim_t<b> batch_internal;
    ducks::gl::make_dim_t<d> depth_internal;
    ducks::gl::make_dim_t<r> rows_internal;
    ducks::gl::make_dim_t<c> cols_internal;

    template <int B=__b__> __attribute__((device)) __attribute__((host)) static constexpr std::enable_if_t<(B > 0), int> batch() { return B; }
    template <int B=__b__> __attribute__((device)) __attribute__((host)) std::enable_if_t<(B == -1), int> batch() const { return batch_internal; }
    template <int D=__d__> __attribute__((device)) __attribute__((host)) static constexpr std::enable_if_t<(D > 0), int> depth() { return D; }
    template <int D=__d__> __attribute__((device)) __attribute__((host)) std::enable_if_t<(D == -1), int> depth() const { return depth_internal; }
    template <int R=__r__> __attribute__((device)) __attribute__((host)) static constexpr std::enable_if_t<(R > 0), int> rows() { return R; }
    template <int R=__r__> __attribute__((device)) __attribute__((host)) std::enable_if_t<(R == -1), int> rows() const { return rows_internal; }
    template <int C=__c__> __attribute__((device)) __attribute__((host)) static constexpr std::enable_if_t<(C > 0), int> cols() { return C; }
    template <int C=__c__> __attribute__((device)) __attribute__((host)) std::enable_if_t<(C == -1), int> cols() const { return cols_internal; }

    detail::descriptor_dict<TMA_Types...> tma_descs;

    __attribute__((host)) inline gl(T *_data,
                        ducks::gl::make_arg_t<b> _batch,
                        ducks::gl::make_arg_t<d> _depth,
                        ducks::gl::make_arg_t<r> _rows,
                        ducks::gl::make_arg_t<c> _cols) :
            raw_ptr(_data), batch_internal(_batch), depth_internal(_depth), rows_internal(_rows), cols_internal(_cols) {
        tma_descs = detail::descriptor_dict<TMA_Types...>(raw_ptr, batch_internal, depth_internal, rows_internal, cols_internal);
    }
    __attribute__((host)) __attribute__((device)) inline gl(const gl &other) :
            raw_ptr(other.raw_ptr), batch_internal(other.batch_internal), depth_internal(other.depth_internal), rows_internal(other.rows_internal), cols_internal(other.cols_internal), tma_descs(other.tma_descs) {}
    __attribute__((device)) inline T& operator[](const coord<ducks::default_type> &idx) const { // yes I am abusing the const qualifier here a bit.
        return raw_ptr[((idx.b*depth() + idx.d)*rows() + idx.r)*cols() + idx.c];
    }
    template<int axis> __attribute__((device)) inline size_t shape() const {
        static_assert(axis==0 || axis==1 || axis==2 || axis==3, "Axis must be 0, 1, 2, or 3.");
        if constexpr (axis==0) { return size_t(batch()); }
        else if constexpr (axis==1) { return size_t(depth()); }
        else if constexpr (axis==2) { return size_t(rows()); }
        else if constexpr (axis==3) { return size_t(cols()); }
    }
    template<int axis> __attribute__((device)) inline size_t stride() const {
        static_assert(axis==0 || axis==1 || axis==2 || axis==3, "Axis must be 0, 1, 2, or 3.");
        if constexpr (axis==0) { return depth()*rows()*cols(); }
        else if constexpr (axis==1) { return rows()*cols(); }
        else if constexpr (axis==2) { return cols(); }
        else if constexpr (axis==3) { return 1; }
    }
};

namespace ducks {
namespace gl {
/**
* @brief Concept for all global layouts.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as ducks::gl::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::gl::identifier
}
}

// Structs for initializing global layouts automatically.
// struct unsafe_gl {
//     uint64_t data;
//     int b, d, r, c;
//     unsafe_gl(uint64_t data, int b, int d, int r, int c) : data(data), b(b), d(d), r(r), c(c) {}
// };
template<int N> auto make_unsafe_gl_arg(int param) { // typename std::conditional_t<(N < 0), std::nullptr_t, int>
    if constexpr (N > 0) { return nullptr; }
    else { return param; }
}
template<ducks::gl::all GL, bool safe=true> __attribute__((host)) inline GL make_gl(uint64_t data, int b, int d, int r, int c) {
    if constexpr (safe) {
        if(GL::__b__ > 0 && b != GL::__b__) {
            throw std::runtime_error("Batch dimension mismatch.");
        }
        if(GL::__d__ > 0 && d != GL::__d__) {
            throw std::runtime_error("Depth dimension mismatch.");
        }
        if(GL::__r__ > 0 && r != GL::__r__) {
            throw std::runtime_error("Row dimension mismatch.");
        }
        if(GL::__c__ > 0 && c != GL::__c__) {
            throw std::runtime_error("Column dimension mismatch.");
        }
    }
    return GL(
        reinterpret_cast<typename GL::dtype*>(data),
        make_unsafe_gl_arg<GL::__b__>(b),
        make_unsafe_gl_arg<GL::__d__>(d),
        make_unsafe_gl_arg<GL::__r__>(r),
        make_unsafe_gl_arg<GL::__c__>(c)
    );
}

} // namespace kittens
# 10 "/root/HipKittens//include/types/global/global.cuh" 2
# 11 "/root/HipKittens//include/types/types.cuh" 2

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

namespace kittens {

/**
 * @brief Row vector type alias.
 *
 * This template alias provides a convenient way to refer to the row vector type
 * associated with a given class or type `T`. It assumes that the class `T` has
 * a nested type named `row_vec`.
 *
 * @tparam T The class or type for which the row vector type is defined.
 *
 * Example usage:
 * @code
 * kittens::row_vec<decltype(some_tile)> row_vector;
 * @endcode
 */
template<typename T>
using row_vec = T::row_vec;

/**
 * @brief Column vector type alias.
 *
 * This template alias provides a convenient way to refer to the column vector type
 * associated with a given class or type `T`. It assumes that the class `T` has
 * a nested type named `col_vec`.
 *
 * @tparam T The class or type for which the column vector type is defined.
 *
 * Example usage:
 * @code
 * kittens::col_vec<decltype(some_tile)> col_vector;
 * @endcode
 */
template<typename T>
using col_vec = T::col_vec;

// ^ this code lives here because it applies to both sv and rv types

// register tile layouts
using row_l = ducks::rt_layout::row;
using col_l = ducks::rt_layout::col;

// register vector layouts
using align_l = ducks::rv_layout::align;
using ortho_l = ducks::rv_layout::ortho;
using naive_l = ducks::rv_layout::naive;

}
# 10 "/root/HipKittens//include/kittens.cuh" 2
# 1 "/root/HipKittens//include/ops/ops.cuh" 1
/**
 * @file
 * @brief A collection of all of the operations that ThunderKittens defines.
 */



# 1 "/root/HipKittens//include/ops/warp/warp.cuh" 1
/**
 * @file
 * @brief An aggregate header of all warp (worker) operations defined by ThunderKittens
 */



// no namespace wrapper needed here
// as warp is the default op scope!

# 1 "/root/HipKittens//include/ops/warp/register/register.cuh" 1
/**
 * @file
 * @brief An aggregate header for warp operations on data stored in registers.
 */



# 1 "/root/HipKittens//include/ops/warp/register/tile/tile.cuh" 1
/**
 * @file
 * @brief An aggregate header for warp operations on register tiles.
 */



# 1 "/root/HipKittens//include/ops/warp/register/tile/conversions.cuh" 1
/**
 * @file
 * @brief Conversions between data layouts and types for register tiles.
 */






namespace kittens {

/* ----------  LAYOUT SWAPS  ---------- */

/**
 * @brief Swaps the layout of a register base tile.
 *
 * This function swaps the layout of a register base tile by performing a series of layout swaps
 * on its constituent bf16_2 elements. It is used to change the data layout within a register tile.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the destination register base tile where the result will be stored.
 * @param src[in] Reference to the source register base tile to be swapped.
 */
template<typename T, ducks::rt_layout::all layout>
__attribute__((device)) inline void swap_layout(rt_base<T, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt_base<T, layout> &src) {
    int lane = laneid();

    int block_src_trans = 16 * ((lane % 16) / 4) + 4 * (lane / 16);
    int block_offset = lane % 4;

    T src_tmp[4] = {
        src.data[0].x, src.data[0].y,
        src.data[1].x, src.data[1].y
    };

    T dst_tmp[4];
#pragma unroll
    for(int k = 0; k < 4; k++) {
        if constexpr (std::is_same_v<T, bf16>) {
            dst_tmp[block_offset^k] = __float2bfloat16(__shfl(__bfloat162float(src_tmp[block_offset^k]), block_src_trans + block_offset^k));
        }
        else {
            dst_tmp[block_offset^k] = __shfl(src_tmp[block_offset^k], block_src_trans + block_offset^k);
        }
    }

    dst.data[0].x = dst_tmp[0];
    dst.data[0].y = dst_tmp[1];
    dst.data[1].x = dst_tmp[2];
    dst.data[1].y = dst_tmp[3];
}

/**
 * @brief Swaps the layout of a register tile.
 *
 * This function swaps the layout of a register tile by iterating over its height and width
 * and performing layout swaps on each of its base elements.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the destination register tile where the result will be stored.
 * @param src[in] Reference to the source register tile to be swapped.
 */
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__attribute__((device)) static inline void swap_layout(rt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt<T2, _height, _width, layout> &src) {

#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
            swap_layout(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}

/**
 * @brief Swaps the layout of a register base tile in place.
 *
 * This function swaps the layout of a register base tile in place by casting it to the
 * transposed layout type and then performing the layout swap.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param src[in] Reference to the register base tile to be swapped in place.
 * @return A reference to the swapped register base tile.
 */
template<typename T2, ducks::rt_layout::all layout>
__attribute__((device)) inline rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(const rt_base<T2, layout> &src) {
    rt_base<T2, typename ducks::rt_layout::transpose<layout>::type> &dst = *(rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>*)(&src);
    swap_layout(dst, src);
    return dst;
}

/**
 * @brief Swaps the layout of a register tile in place.
 *
 * This function swaps the layout of a register tile in place by iterating over its height and width
 * and performing in-place layout swaps on each of its base elements.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be swapped in place.
 * @return A reference to the swapped register tile.
 */
template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
__attribute__((device)) static inline rt<T2, _rows, _cols, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(rt<T2, _rows, _cols, layout> &tile) {
#pragma unroll
    for(int i = 0; i < tile.height; i++) {
#pragma unroll
        for(int j = 0; j < tile.width; j++) {
            swap_layout_inplace(tile.tiles[i][j]);
        }
    }
    return *(rt<T2, _rows, _cols, typename ducks::rt_layout::transpose<layout>::type>*)(&tile);
}

/* ----------  TRANSPOSE  ---------- */

/**
 * @brief Transposes a register base tile.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the register tile in which to store the transposed src.
 * @param src[in] Reference to the register base tile to be transposed.
 */
template<typename T, ducks::rt_layout::all layout>
__attribute__((device)) inline void transpose(rt_base<T, layout> &dst, const rt_base<T, layout> &src) {
    int lane = laneid();
    int block_src_trans = 16*((lane%16)/4) + 4*(lane/16);
    int block_offset = lane%4;

    T src_tmp[4] = {
        src.data[0].x, src.data[0].y,
        src.data[1].x, src.data[1].y
    };

    T dst_tmp[4];

#pragma unroll
    for(int k = 0; k < 4; k++) {
        if constexpr (std::is_same_v<T, bf16>) {
            dst_tmp[block_offset^k] = __float2bfloat16(__shfl(__bfloat162float(src_tmp[block_offset^k]), block_src_trans + block_offset^k));
        }
        else {
            dst_tmp[block_offset^k] = __shfl(src_tmp[block_offset^k], block_src_trans + block_offset^k);
        }
    }

    dst.data[0].x = dst_tmp[0];
    dst.data[0].y = dst_tmp[1];
    dst.data[1].x = dst_tmp[2];
    dst.data[1].y = dst_tmp[3];
}
/**
 * @brief Transposes a register tile.
 * 
 * This function is marked "sep", which means that the registers underlying dst MUST be separate
 * from the registers underlying src.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the src register tile, and the width of the dst tile.
 * @tparam _width The width of the src register tile, and the height of the dst tile.
 * @tparam layout The layout of the register tile.
 * @param dst[out] Reference to the register tile in which to store the transposed src.
 * @param src[in] Reference to the register tile to be transposed.
 */
template<ducks::rt::all RT>
__attribute__((device)) static inline void transpose_sep(RT &dst, const rt<typename RT::T, RT::cols, RT::rows, typename RT::layout> &src) {
#pragma unroll
    for(int i = 0; i < RT::height; i++) {
#pragma unroll
        for(int j = 0; j < RT::width; j++) {
            transpose(dst.tiles[i][j], src.tiles[j][i]);
        }
    }
}

/**
 * @brief Transposes a register base tile in-place.
 *
 * @tparam T2 The data type of the register base tile elements.
 * @tparam layout The current layout of the register base tile.
 * @param src[in] Reference to the register tile to be transposed.
 * @return A reference to the transposed register base tile.
 */
template<typename T2, ducks::rt_layout::all layout>
__attribute__((device)) inline rt_base<T2, layout>& transpose_inplace(rt_base<T2, layout> &src) {
    transpose(src, src);
    return src;
}
/**
 * @brief Transposes a square register tile in-place.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height (in units of 16) of the src register tile, and the width of the dst tile. (Must be the same as _width.)
 * @tparam _width The width (in units of 16) of the src register tile, and the height of the dst tile. (Must be the same as _height.)
 * @tparam layout The current layout of the register tile.
 * @param src[in] Reference to the register tile to be transposed.
 * @return A reference to the transposed register tile.
 */
template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
__attribute__((device)) static inline rt<T2, _rows, _cols, layout>& transpose_inplace(rt<T2, _rows, _cols, layout> &tile) {
    static_assert(_cols == _rows, "in-place register tile transpose is only allowed for square tiles.");
#pragma unroll
    for(int i = 0; i < tile.height; i++) {
#pragma unroll
        for(int j = 0; j < i; j++) {
            rt_base<T2, layout> tmp;
            copy(tmp, tile.tiles[i][j]);
            transpose(tile.tiles[i][j], tile.tiles[j][i]);
            transpose(tile.tiles[j][i], tmp);
        }
        transpose_inplace(tile.tiles[i][i]);
    }
    return tile;
}

/* ----------  TYPE SWAPS  ---------- */

/**
 * @brief Copies a register base tile, converting the underlying type if necessary.
 *
 * @tparam T2 The data type of the destination register elements.
 * @tparam U2 The data type of the source register elements.
 * @tparam layout The current layout of the register base tile.
 * @param[out] dst A reference to the destination register base tile.
 * @param[in] src A reference to the source register base tile.
 */
template<typename T, typename U, ducks::rt_layout::all layout>
__attribute__((device)) static inline void copy(rt_base<T, layout> &dst, const rt_base<U, layout> &src) {
    using T2 = typename base_types::packing<T>::packed_type;
    using U2 = typename base_types::packing<U>::packed_type;
#pragma unroll
    for(int k = 0; k < dst.packed_per_thread; k++) {
        dst.data[k] = base_types::convertor<T2, U2>::convert(src.data[k]);
    }
}

/**
 * @brief Copies a register tile, converting the underlying type if necessary.
 *
 * @tparam T2 The data type of the destination register elements.
 * @tparam U2 The data type of the source register elements.
 * @tparam _height The height (in units of 16) of the register tiles.
 * @tparam _width The width (in units of 16) of the register tiles.
 * @tparam layout The current layout of the register tile.
 * @param[out] dst A reference to the destination register tile.
 * @param[in] src A reference to the source register tile.
 */
template<typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
__attribute__((device)) static inline void copy(rt<T2, _height, _width, layout> &dst, const rt<U2, _height, _width, layout> &src) {
#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
            copy(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}

/* ----------  CAUSAL  ---------- */

/**
 * @brief Makes a square register tile causal by zeroing elements above the main diagonal.
 *
 * This function modifies a square register tile in-place to make it causal. All elements
 * above the main diagonal are set to zero, while elements on or below the main diagonal
 * are left unchanged.
 *
 * @tparam T The data type of the register tile elements.
 * @tparam _size The size (height and width) of the square register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be made causal.
 */
template<ducks::rt::row_layout RT>
__attribute__((device)) static inline void make_causal(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    int lane = laneid();
#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if(j < i) { // below the diagonal, copy
#pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j > i) { // above the diagonal, zero
#pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else { // on the diagonal, interesting!
                constexpr uint64_t MASKS[4] = {0xF000FF00FFF0FFFF, 0xE000FE00FFE0FFFE,
                                               0xC000FC00FFC0FFFC, 0x8000F800FF80FFF8}; // magic numbers for on-diagonal core matrices

#pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    if ((MASKS[k * 2] >> lane) & 1) {
                        dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x;
                    }
                    else {
                        dst.tiles[i][j].data[k].x = val;
                    }
                    if ((MASKS[k * 2 + 1] >> laneid()) & 1) {
                        dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y;
                    }
                    else {
                        dst.tiles[i][j].data[k].y = val;
                    }
                }
            }
            // __syncwarp();
        }
    }
}

/**
 * @brief Makes a square register tile anti-causal by zeroing elements below the main diagonal.
 *
 * This function modifies a square register tile in-place to make it anti-causal. All elements
 * below the main diagonal are set to zero, while elements on or above the main diagonal
 * are left unchanged.
 *
 * @tparam T The data type of the register tile elements.
 * @tparam _size The size (height and width) of the square register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be made causal.
 */
template<ducks::rt::row_layout RT>
__attribute__((device)) static inline void make_causal_t(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if(j > i) { // above the diagonal, copy
#pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j < i) { // below the diagonal, zero
#pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else { // on the diagonal, interesting!
                constexpr uint64_t MASKS[4] = {0x1FFF01FF001F0001, 0x3FFF03FF003F0003,
                                               0x7FFF07FF007F0007, 0xFFFF0FFF00FF000F}; // magic numbers for on-diagonal core matrices

#pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    if ((MASKS[k * 2] >> laneid()) & 1) {
                        dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x;
                    }
                    else {
                        dst.tiles[i][j].data[k].x = val;
                    }
                    if ((MASKS[k * 2 + 1] >> laneid()) & 1) {
                        dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y;
                    }
                    else {
                        dst.tiles[i][j].data[k].y = val;
                    }
                }

            }
            // __syncwarp();
        }
    }
}

/* ----------  TRIANGULAR FILLS  ---------- */

/**
 * @brief Makes a register tile triangular by zeroing elements above the row index
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to triangularize from.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__attribute__((device)) static inline void tril(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    const int lane = laneid();
    const int row = lane % 16;
    const int col = 4 * (lane / 16);
#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int global_row_idx = (i * dst.tile_size_row) + row;
                const int global_col_idx_x = (j * dst.tile_size_col) + col + 2 * k;
                const int global_col_idx_y = (j * dst.tile_size_col) + col + 2 * k + 1;

                if (global_row_idx < row_idx) { dst.tiles[i][j].data[k] = packed_val; }
                else {
                    if (global_col_idx_x <= global_row_idx - row_idx) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                    else { dst.tiles[i][j].data[k].x = val; }

                    if (global_col_idx_y <= global_row_idx - row_idx) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                    else { dst.tiles[i][j].data[k].y = val; }
                }
            }
        }
        // __syncwarp();
    }
}


template<ducks::rt::col_layout RT>
__attribute__((device)) static inline void tril(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {

    const int lane = laneid();
    const int row = 4 * (lane / 16);
    const int col = lane % 16;

#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int global_row_idx_x = (i * dst.tile_size_row) + row + 2 * k;
                const int global_row_idx_y = (i * dst.tile_size_row) + row + 2 * k + 1;
                const int global_col_idx = (j * dst.tile_size_col) + col;

                if (global_row_idx_x < row_idx) { dst.tiles[i][j].data[k].x = val; }
                else {
                    if (global_col_idx <= global_row_idx_x - row_idx) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                    else { dst.tiles[i][j].data[k].x = val; }
                }

                if (global_row_idx_y < row_idx) { dst.tiles[i][j].data[k].y = val; }
                else {
                    if (global_col_idx <= global_row_idx_y - row_idx) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                    else { dst.tiles[i][j].data[k].y = val; }
                }
            }
        }
        // __syncwarp();
    }
}

/**
 * @brief Makes a register tile triangular by zeroing elements below the row index
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to triangularize from.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__attribute__((device)) static inline void triu(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    const int lane = laneid();
    const int row = lane % 16;
    const int col = 4 * (lane / 16);
#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int global_row_idx = (i * dst.tile_size_row) + row;
                const int global_col_idx_x = (j * dst.tile_size_col) + col + 2 * k;
                const int global_col_idx_y = (j * dst.tile_size_col) + col + 2 * k + 1;

                if (global_row_idx < row_idx) { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
                else {
                    if (global_col_idx_x < global_row_idx - row_idx) { dst.tiles[i][j].data[k].x = val; }
                    else { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }

                    if (global_col_idx_y < global_row_idx - row_idx) { dst.tiles[i][j].data[k].y = val; }
                    else { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                }
            }
        }
        // __syncwarp();
    }
}


template<ducks::rt::col_layout RT>
__attribute__((device)) static inline void triu(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {

    const int lane = laneid();
    const int row = 4 * (lane / 16);
    const int col = lane % 16;

#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int global_row_idx_x = (i * dst.tile_size_row) + row + 2 * k;
                const int global_row_idx_y = (i * dst.tile_size_row) + row + 2 * k + 1;
                const int global_col_idx = (j * dst.tile_size_col) + col;

                if (global_row_idx_x < row_idx) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                else {
                    if (global_col_idx < global_row_idx_x - row_idx) { dst.tiles[i][j].data[k].x = val; }
                    else { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                }

                if (global_row_idx_y < row_idx) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                else {
                    if (global_col_idx < global_row_idx_y - row_idx) { dst.tiles[i][j].data[k].y = val; }
                    else { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                }
            }
        }
        // __syncwarp();
    }
}

/* ----------  RECTANGULAR FILLS  ---------- */

/**
 * @brief Makes a register tile right filled with a given value.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param col_idx[in] The column index to fill from and onwards to the right.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__attribute__((device)) static inline void right_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(col_idx >= dst.cols) return;

    const int col = 4 * (laneid() / 16);
#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int col_idx_x = (j * dst.tile_size_col) + col + 2 * k;
                const int col_idx_y = (j * dst.tile_size_col) + col + 2 * k + 1;
                if (col_idx_x >= col_idx) { dst.tiles[i][j].data[k].x = val; }
                else { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (col_idx_y >= col_idx) { dst.tiles[i][j].data[k].y = val; }
                else { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}


template<ducks::rt::col_layout RT>
__attribute__((device)) static inline void right_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    const int col = laneid() % 16;
#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int t_col_idx = (j * dst.tile_size_col) + col;
                if (t_col_idx >= col_idx) { dst.tiles[i][j].data[k] = packed_val; }
                else { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
        // __syncwarp();
    }
}

/**
 * @brief Makes a register tile left filled with a given value.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param col_idx[in] The column index to fill to the left (exclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__attribute__((device)) static inline void left_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(col_idx <= 0) return;

    const int col = 4 * (laneid() / 16);
#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int col_idx_x = (j * dst.tile_size_col) + col + 2 * k;
                const int col_idx_y = (j * dst.tile_size_col) + col + 2 * k + 1;
                if (col_idx_x < col_idx) { dst.tiles[i][j].data[k].x = val; }
                else { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (col_idx_y < col_idx) { dst.tiles[i][j].data[k].y = val; }
                else { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}


template<ducks::rt::col_layout RT>
__attribute__((device)) static inline void left_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    const int col = laneid() % 16;
#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int thread_col = (j * dst.tile_size_col) + col;
                if (thread_col < col_idx) { dst.tiles[i][j].data[k] = packed_val; }
                else { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
        // __syncwarp();
    }
}

/**
 * @brief Makes a register tile upper filled with a given value.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to fill to, from the top (exclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__attribute__((device)) static inline void upper_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(row_idx <= 0) return;
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    const int row = laneid() % 16;
#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int thread_row = (i * dst.tile_size_row) + row;
                if (thread_row < row_idx) { dst.tiles[i][j].data[k] = packed_val; }
                else { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
    }
}


template<ducks::rt::col_layout RT>
__attribute__((device)) static inline void upper_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const int row = 4 * (laneid() / 16);
#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int row_idx_x = (i * dst.tile_size_row) + row + 2 * k;
                const int row_idx_y = (i * dst.tile_size_row) + row + 2 * k + 1;
                if (row_idx_x < row_idx) { dst.tiles[i][j].data[k].x = val; }
                else { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (row_idx_y < row_idx) { dst.tiles[i][j].data[k].y = val; }
                else { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}

/**
 * @brief Makes a register tile lower filled with a given value.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to fill from and onwards to the bottom of the tile (inclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__attribute__((device)) static inline void lower_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(row_idx >= dst.rows) return;
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    const int row = laneid() % 16;
#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int thread_row = (i * dst.tile_size_row) + row;
                if (thread_row >= row_idx) { dst.tiles[i][j].data[k] = packed_val; }
                else { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
    }
}

template<ducks::rt::col_layout RT>
__attribute__((device)) static inline void lower_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const int row = 4 * (laneid() / 16);
#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int row_idx_x = (i * dst.tile_size_row) + row + 2 * k;
                const int row_idx_y = (i * dst.tile_size_row) + row + 2 * k + 1;
                if (row_idx_x >= row_idx) { dst.tiles[i][j].data[k].x = val; }
                else { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (row_idx_y >= row_idx) { dst.tiles[i][j].data[k].y = val; }
                else { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}

/* ----------  SUBTILE  ---------- */

/**
* @brief Returns a reference to a subtile of the given tile.
*
* @tparam subtile_height The height of the subtile.
* @tparam RT The type of the input tile, which must satisfy the ducks::rt::all concept.
* @param src The input tile.
* @param idx The coord of the subtile.
* @return A reference to the subtile.
*
* @note The subtile height must evenly divide the tile height.
*/
template<int subtile_rows, ducks::rt::all RT>
__attribute__((device)) inline rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout> &subtile_inplace(RT & src, int idx) {
    using T = typename RT::T;
    static_assert(RT::height % (subtile_rows / TILE_ROW_DIM<T>) == 0, "subtile height should evenly divide tile height.");
    return reinterpret_cast<rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout>&>(
        src.tiles[idx*(subtile_rows / TILE_ROW_DIM<T>)]
    );
}

}
# 9 "/root/HipKittens//include/ops/warp/register/tile/tile.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/register/tile/maps.cuh" 1
/**
 * @file
 * @brief Map operations: between tiles, and those which apply vectors to tiles.
 */






namespace kittens {

/* ----------  Uniform tile maps (independent of layout)  ---------- */

/**
 * @brief Applies a unary operation to each element of a tile.
 *
 * @tparam op Unary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 */
template<typename op, ducks::rt::all T>
__attribute__((device)) static inline void unary_map(T &dst, const T &src) {
#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(src.tiles[i][j].data[k]);
            }
        }
    }
}

/**
 * @brief Applies a binary operation to each element of a tile with a scalar parameter.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param param[in] Scalar parameter for the binary operation.
 */
template<typename op, ducks::rt::all T>
__attribute__((device)) static inline void bin_map(T &dst, const T &src, const typename T::dtype &param) {
#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(src.tiles[i][j].data[k], param);
            }
        }
    }
}
/**
 * @brief Applies a binary operation to each element of a tile with an unpacked scalar parameter.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param param[in] Unpacked scalar parameter for the binary operation.
 */
template<typename op, ducks::rt::all T>
__attribute__((device)) static inline void bin_map(T &dst, const T &src, const typename base_types::packing<typename T::dtype>::unpacked_type &param) {
    // The optimizing compiler should eliminate this pack in the 32-bit case but not in the 16-bit case
    bin_map<op, T>(dst, src, base_types::packing<typename T::dtype>::pack(param));
}
/**
 * @brief Applies a binary operation element-wise between two tiles.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the operation.
 * @param rhs[in] Right-hand side source tile for the operation.
 */
template<typename op, ducks::rt::all T>
__attribute__((device)) static inline void bin_map(T &dst, const T &lhs, const T &rhs) {
#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(lhs.tiles[i][j].data[k], rhs.tiles[i][j].data[k]);
            }
        }
    }
}

/* ----------  Row tile maps  ----------*/

/**
 * @brief Applies an operation across the rows of a tile in a row-major layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param row_values[in] Column vector containing values to apply across each row.
 */
template<typename op, ducks::rt::row_layout T, ducks::rv::all V>
__attribute__((device)) static inline void row_map(T &dst, const T &src, const V &row_values) {

    using dtype = T::dtype;
    using RT = V::dtype;
    using RT2 = base_types::packing<RT>::packed_type;

    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>); // compatible layout
    static_assert(std::is_same_v<RT2, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::height); // compatible size

#pragma unroll
    for(int i = 0; i < dst.height; i++) {
        RT packed_val = row_values[i][0]; //  first value in eager mode
        RT2 packed_val2 = {packed_val, packed_val};
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<dtype>(src.tiles[i][j].data[k], packed_val2);
            }
        }
    }
}
/**
 * @brief Applies an operation across the rows of a tile in a column-major layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with column-major layout.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param row_values[in] Column vector containing values to apply across each row.
 */
template<typename op, ducks::rt::col_layout T, ducks::rv::all V>
__attribute__((device)) static inline void row_map(T &dst, const T &src, const V &row_values) {

    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = T::dtype;

#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<dtype>(src.tiles[i][j].data[k], row_values[i][k]);
            }
        }
    }
}

// Three-operand row map. Mostly useful for FMA instructions.

/**
 * @brief Applies an operation across the rows of two tiles in a row-major layout, using a third operand.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param a[in] First source tile to apply the operation on.
 * @param b[in] Second source tile to apply the operation on.
 * @param row_values[in] Column vector containing values to apply across each row.
 */
template<typename op, ducks::rt::row_layout T, ducks::rv::all V>
__attribute__((device)) static inline void row_map(T &dst, const T &a, const T &b, const V &row_values) {

    using dtype = T::dtype;
    using RT = V::dtype;
    using RT2 = base_types::packing<RT>::packed_type;

    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>); // compatible layout
    static_assert(std::is_same_v<RT2, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::height); // compatible size

#pragma unroll
    for(int i = 0; i < dst.height; i++) {
        dtype packed_val = base_types::packing<dtype>::pack(row_values[i][0]); //  first value in eager mode
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<dtype>(a.tiles[i][j].data[k], b.tiles[i][j].data[k], packed_val);
            }
        }
    }
}
/**
 * @brief Applies an operation across the rows of two tiles in a column-major layout, using a third operand.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with column-major layout.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param a[in] First source tile to apply the operation on.
 * @param b[in] Second source tile to apply the operation on.
 * @param row_values[in] Column vector containing values to apply across each row.
 */
template<typename op, ducks::rt::col_layout T, ducks::rv::all V>
__attribute__((device)) static inline void row_map(T &dst, const T &a, const T &b, const V &row_values) {

    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = T::dtype;

#pragma unroll
    for(int i = 0; i < dst.height; i++) {
#pragma unroll
        for(int j = 0; j < dst.width; j++) {
#pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<dtype>(a.tiles[i][j].data[k], b.tiles[i][j].data[k], row_values[i][k]);
            }
        }
    }
}

/* ----------  Col major tile maps  ----------*/

/**
 * @brief Applies an operation across the columns of a tile in a row-major layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param col_values[in] Row vector containing values to apply across each column.
 */
template<typename op, ducks::rt::row_layout T, ducks::rv::all V>
__attribute__((device)) static inline void col_map(T &dst, const T &src, const V &col_values) {

    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::width); // compatible size

    using dtype = T::dtype;

#pragma unroll
    for(int j = 0; j < dst.width; j++) {
#pragma unroll
        for(int i = 0; i < dst.height; i++) {
#pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<dtype>(src.tiles[i][j].data[k], col_values[j][k]);
            }
        }
    }
}
/**
 * @brief Applies an operation across the columns of a tile in a column-major layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with column-major layout.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param col_values[in] Row vector containing values to apply across each column.
 */
template<typename op, ducks::rt::col_layout T, ducks::rv::all V>
__attribute__((device)) static inline void col_map(T &dst, const T &src, const V &col_values) {

    using dtype = T::dtype;
    using RT = V::dtype;
    using RT2 = base_types::packing<RT>::packed_type;

    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<RT2, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::width); // compatible size

#pragma unroll
    for(int j = 0; j < dst.width; j++) {
        RT packed_val = col_values[j][0]; //  first value in eager mode
        RT2 packed_val2 = {packed_val, packed_val};
#pragma unroll
        for(int i = 0; i < dst.height; i++) {
#pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<dtype>(src.tiles[i][j].data[k], packed_val2);
            }
        }
    }
}

// Three-operand col map
/**
 * @brief Applies an operation across the columns of two tiles in a row-major layout, using a third operand.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param a[in] First source tile to apply the operation on.
 * @param b[in] Second source tile to apply the operation on.
 * @param col_values[in] Row vector containing values to apply across each column.
 */
template<typename op, ducks::rt::row_layout T, ducks::rv::all V>
__attribute__((device)) static inline void col_map(T &dst, const T &a, const T &b, const V &col_values) {

    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::width); // compatible size

    using dtype = T::dtype;

#pragma unroll
    for(int j = 0; j < dst.width; j++) {
#pragma unroll
        for(int i = 0; i < dst.height; i++) {
#pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<dtype>(a.tiles[i][j].data[k], b.tiles[i][j].data[k], col_values[j][k]);
            }
        }
    }
}
/**
 * @brief Applies an operation across the columns of two tiles in a column-major layout, using a third operand.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with column-major layout.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param a[in] First source tile to apply the operation on.
 * @param b[in] Second source tile to apply the operation on.
 * @param col_values[in] Row vector containing values to apply across each column.
 */
template<typename op, ducks::rt::col_layout T, ducks::rv::all V>
__attribute__((device)) static inline void col_map(T &dst, const T &a, const T &b, const V &col_values) {

    using dtype = T::dtype;
    using RT = V::dtype;
    using RT2 = base_types::packing<RT>::packed_type;

    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<RT2, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::width); // compatible size

#pragma unroll
    for(int j = 0; j < dst.width; j++) {
        dtype packed_val = base_types::packing<dtype>::pack(col_values[j][0]); //  first value in eager mode
#pragma unroll
        for(int i = 0; i < dst.height; i++) {
#pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<dtype>(a.tiles[i][j].data[k], b.tiles[i][j].data[k], packed_val);
            }
        }
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// All of the annoying qualifiers *should* be automatically inferred during compile-time.
// So, syntax should just be kittens::add_row(tile, colvec);

/**
 * @brief Sets all elements of a tile to zero.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<ducks::rt::all T>
__attribute__((device)) static inline void zero(T &dst) {
    unary_map<base_ops::zero, T>(dst, dst);
}
/**
 * @brief Sets all elements of a tile to one.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<ducks::rt::all T>
__attribute__((device)) static inline void one(T &dst) {
    unary_map<base_ops::one, T>(dst, dst);
}
/**
 * @brief Sets all elements of a tile to positive infinity.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<ducks::rt::all T>
__attribute__((device)) static inline void pos_infty(T &dst) {
    unary_map<base_ops::pos_infty, T>(dst, dst);
}
/**
 * @brief Sets all elements of a tile to negative infinity.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<ducks::rt::all T>
__attribute__((device)) static inline void neg_infty(T &dst) {
    unary_map<base_ops::neg_infty, T>(dst, dst);
}

/**
 * @brief Applies the exponential function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the exponential function on.
 */
template<ducks::rt::all T>
__attribute__((device)) static inline void exp(T &dst, const T &src) {
    unary_map<base_ops::exp, T>(dst, src);
}
/**
 * @brief Applies the exponential function to each element of a tile, in base 2.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the exponential function on.
 */
template<ducks::rt::all T>
__attribute__((device)) static inline void exp2(T &dst, const T &src) {
    unary_map<base_ops::exp2, T>(dst, src);
}
/**
 * @brief Applies the natural logarithm function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the natural logarithm function on.
 */
template<ducks::rt::all T>
__attribute__((device)) static inline void log(T &dst, const T &src) {
    unary_map<base_ops::log, T>(dst, src);
}
/**
 * @brief Applies the logarithm base 2 function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the logarithm base 2 function on.
 */
template<ducks::rt::all T>
__attribute__((device)) static inline void log2(T &dst, const T &src) {
    unary_map<base_ops::log2, T>(dst, src);
}
/**
 * @brief Applies the absolute value function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the absolute value function on.
 */
template<ducks::rt::all T>
__attribute__((device)) static inline void abs(T &dst, const T &src) {
    unary_map<base_ops::abs, T>(dst, src);
}
/**
 * @brief Applies the rectified linear unit (ReLU) function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the ReLU function on.
 */
template<ducks::rt::all T>
__attribute__((device)) static inline void relu(T &dst, const T &src) {
    unary_map<base_ops::relu, T>(dst, src);
}
/**
 * @brief Copies the elements from one tile to another.
 *
 * @tparam T Destination tile type.
 * @tparam U Source tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to copy from.
 */
template<ducks::rt::all T, typename U>
__attribute__((device)) static inline void copy(T &dst, const U &src) {
    bin_map<base_ops::copy2, T>(dst, src);
}

/**
 * @brief Applies the max operation element-wise between two tiles or a tile and a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the operation.
 * @param rhs[in] Right-hand side source tile or scalar for the operation.
 */
template<ducks::rt::all T, typename U>
__attribute__((device)) static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::max, T>(dst, lhs, rhs);
}
/**
 * @brief Applies the min operation element-wise between two tiles or a tile and a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the operation.
 * @param rhs[in] Right-hand side source tile or scalar for the operation.
 */
template<ducks::rt::all T, typename U>
__attribute__((device)) static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::min, T>(dst, lhs, rhs);
}
/**
 * @brief Adds two tiles element-wise or adds a scalar to each element of a tile.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the addition.
 * @param rhs[in] Right-hand side source tile or scalar for the addition.
 */
template<ducks::rt::all T, typename U>
__attribute__((device)) static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sum, T>(dst, lhs, rhs);
}
/**
 * @brief Subtracts two tiles element-wise or subtracts a scalar from each element of a tile.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the subtraction.
 * @param rhs[in] Right-hand side source tile or scalar for the subtraction.
 */
template<ducks::rt::all T, typename U>
__attribute__((device)) static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sub, T>(dst, lhs, rhs);
}
/**
 * @brief Multiplies two tiles element-wise or multiplies each element of a tile by a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the multiplication.
 * @param rhs[in] Right-hand side source tile or scalar for the multiplication.
 */
template<ducks::rt::all T, typename U>
__attribute__((device)) static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::mul, T>(dst, lhs, rhs);
}
/**
 * @brief Divides two tiles element-wise or divides each element of a tile by a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the division.
 * @param rhs[in] Right-hand side source tile or scalar for the division.
 */
template<ducks::rt::all T, typename U>
__attribute__((device)) static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::div, T>(dst, lhs, rhs);
}

/**
 * @brief Adds row values to each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the addition on.
 * @param row_values[in] Column vector containing values to add to each row.
 */
template<ducks::rt::all T, ducks::rv::all V>
__attribute__((device)) static inline void add_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sum, T, V>(dst, src, row_values);
}
/**
 * @brief Subtracts row values from each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the subtraction on.
 * @param row_values[in] Column vector containing values to subtract from each row.
 */
template<ducks::rt::all T, ducks::rv::all V>
__attribute__((device)) static inline void sub_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sub, T, V>(dst, src, row_values);
}
/**
 * @brief Multiplies each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the multiplication on.
 * @param row_values[in] Column vector containing values to multiply each row by.
 */
template<ducks::rt::all T, ducks::rv::all V>
__attribute__((device)) static inline void mul_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::mul, T, V>(dst, src, row_values);
}
/**
 * @brief Divides each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the division on.
 * @param row_values[in] Column vector containing values to divide each row by.
 */
template<ducks::rt::all T, ducks::rv::all V>
__attribute__((device)) static inline void div_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::div, T, V>(dst, src, row_values);
}
/**
 * @brief Broadcast a vector into into a tile's rows.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param row_values[in] Column vector containing values to broadcast into rows.
 */
template<ducks::rt::all T, ducks::rv::all V>
__attribute__((device)) static inline void broadcast_row(T &dst, const V &row_values) {
    row_map<base_ops::copy2, T, V>(dst, dst, row_values);
}


// col maps
/**
 * @brief Adds column values to each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the addition on.
 * @param col_values[in] Row vector containing values to add to each column.
 */
template<ducks::rt::all T, ducks::rv::all V>
__attribute__((device)) static inline void add_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sum, T, V>(dst, src, col_values);
}
/**
 * @brief Subtracts column values from each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the subtraction on.
 * @param col_values[in] Row vector containing values to subtract from each column.
 */
template<ducks::rt::all T, ducks::rv::all V>
__attribute__((device)) static inline void sub_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sub, T, V>(dst, src, col_values);
}
/**
 * @brief Multiplies each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the multiplication on.
 * @param col_values[in] Row vector containing values to multiply each column by.
 */
template<ducks::rt::all T, ducks::rv::all V>
__attribute__((device)) static inline void mul_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::mul, T, V>(dst, src, col_values);
}
/**
 * @brief Divides each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the division on.
 * @param col_values[in] Row vector containing values to divide each column by.
 */
template<ducks::rt::all T, ducks::rv::all V>
__attribute__((device)) static inline void div_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::div, T, V>(dst, src, col_values);
}
/**
 * @brief Broadcast a vector into into a tile's columns.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param row_values[in] Row vector containing values to broadcast into cols.
 */
template<ducks::rt::all T, ducks::rv::all V>
__attribute__((device)) static inline void broadcast_col(T &dst, const V &col_values) {
    col_map<base_ops::copy2, T, V>(dst, dst, col_values);
}

}
# 10 "/root/HipKittens//include/ops/warp/register/tile/tile.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/register/tile/reductions.cuh" 1
/**
 * @file
 * @brief Reduction operations mapping tiles to vectors.
 */






namespace kittens {

/**
 * @brief Perform a row-wise reduction on a matrix in row-major layout.
 *
 * This function template performs a parallel reduction across the rows of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type with row layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */
template<typename op, ducks::rv::all V, ducks::rt::row_layout T, bool reset>
__attribute__((device)) static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    using dtype = T::dtype;
    using RT2 = V::dtype;
    using RT = base_types::packing<RT2>::unpacked_type;

    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename base_types::packing<RT2>::packed_type, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::height); // compatible size

    const int leader = laneid() % 16;

#pragma unroll
    for(int i = 0; i < src.height; i++) {
        dtype accum_packed = src.tiles[i][0].data[0];
        for (int k = 1; k < src.packed_per_tile; k++) {
            accum_packed = op::template op<dtype>(accum_packed, src.tiles[i][0].data[k]);
        }

#pragma unroll
        for(int j = 1; j < src.width; j++) {
#pragma unroll
            for (int k = 0; k < src.packed_per_tile; k++) {
                accum_packed = op::template op<dtype>(accum_packed, src.tiles[i][j].data[k]);
            }
        }
        RT accum_single = op::template op<RT>(accum_packed.x, accum_packed.y);
        // Now we need to do a lil shuffle to make everyone happy.

        accum_single = op::template op<RT>(accum_single, packed_shfl_down(MASK_ALL, accum_single, 32));
        accum_single = op::template op<RT>(accum_single, packed_shfl_down(MASK_ALL, accum_single, 16));

        if(reset) {
            row_accum[i][0] = accum_single;
        }
        else {
            row_accum[i][0] = op::template op<RT>(src_accum[i][0], accum_single);
        }

        row_accum[i][0] = packed_shfl(MASK_ALL, row_accum[i][0], leader);
        row_accum[i][0] = row_accum[i][0];
    }
}

/**
 * @brief Perform a row-wise reduction on a matrix in column-major layout.
 *
 * This function template performs a parallel reduction across the rows of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication and is optimized for column-major matrices.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type with column layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */
template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
__attribute__((device)) static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::height); // compatible size

    using RT2 = V::dtype;
    using RT = base_types::packing<RT2>::unpacked_type;

    const int leader = (laneid() / 16) * 16;
    const int packed_per_tile = 2;
    const int max_shift = 8;

    RT2 accum[packed_per_tile];

#pragma unroll
    for(int i = 0; i < src.height; i++) {
#pragma unroll
        for(int k = 0; k < packed_per_tile; k++) {
            accum[k] = src.tiles[i][0].data[k];
        }
#pragma unroll
        for(int j = 1; j < src.width; j++) {
#pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                accum[k] = op::template op<RT2>(accum[k], src.tiles[i][j].data[k]);
            }
        }

#pragma unroll
        for(int k = 0; k < packed_per_tile; k++) {
            for (int shift = max_shift; shift > 0; shift /= 2) {
                accum[k] = op::template op<RT2>(accum[k], packed_shfl_down(MASK_ALL, accum[k], shift));
            }
        }

        if(reset) {
#pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                row_accum[i][k] = accum[k];
            }
        }
        else {
#pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                row_accum[i][k] = op::template op<RT2>(src_accum[i][k], accum[k]);
            }
        }

#pragma unroll
        for(int k = 0; k < packed_per_tile; k++) {
            row_accum[i][k] = packed_shfl(MASK_ALL, row_accum[i][k], leader);
        }
    }
}

// Col reduction.
/**
 * @brief Perform a column-wise reduction on a matrix in row-major layout.
 *
 * This function template performs a parallel reduction across the columns of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication and is optimized for row-major matrices.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the column accumulator.
 * @tparam T The matrix type with row layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */
template<typename op, ducks::rv::all V, ducks::rt::row_layout T, bool reset>
__attribute__((device)) static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::width); // compatible size

    using RT2 = V::dtype;
    using RT = base_types::packing<RT2>::unpacked_type;

    const int leader = (laneid() / 16) * 16;
    const int packed_per_tile = 2;
    const int max_shift = 8;

    RT2 accum[packed_per_tile];

#pragma unroll
    for(int j = 0; j < src.width; j++) {
#pragma unroll
        for(int k = 0; k < packed_per_tile; k++) {
            accum[k] = src.tiles[0][j].data[k];
        }
#pragma unroll
        for(int i = 1; i < src.height; i++) {
#pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                accum[k] = op::template op<RT2>(accum[k], src.tiles[i][j].data[k]);
            }
        }

#pragma unroll
        for(int k = 0; k < packed_per_tile; k++) {
            for (int shift = max_shift; shift > 0; shift /= 2) {
                accum[k] = op::template op<RT2>(accum[k], packed_shfl_down(MASK_ALL, accum[k], shift));
            }
        }

        if(reset) {
#pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                col_accum[j][k] = accum[k];
            }
        }
        else {
#pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                col_accum[j][k] = op::template op<RT2>(src_accum[j][k], accum[k]);
            }
        }

#pragma unroll
        for(int k = 0; k < packed_per_tile; k++) {
            col_accum[j][k] = packed_shfl(MASK_ALL, col_accum[j][k], leader);
        }
    }
}
/**
 * @brief Perform a column-wise reduction on a matrix in column-major layout.
 *
 * This function template performs a parallel reduction across the columns of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication and is optimized for column-major matrices.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the column accumulator.
 * @tparam T The matrix type with column layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */
template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
__attribute__((device)) static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    using RT2 = base_types::packing<typename V::dtype>::packed_type;
    using RT = base_types::packing<RT2>::unpacked_type;

    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<RT2, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::width); // compatible size

    const int leader = laneid() % 16;
#pragma unroll
    for(int j = 0; j < src.width; j++) { // note now width is the outer loop
        RT2 accum_packed = op::template op<RT2>(src.tiles[0][j].data[0], src.tiles[0][j].data[1]);
#pragma unroll
        for(int i = 1; i < src.height; i++) { // and height is the inner loop
#pragma unroll
            for(int k = 0; k < src.packed_per_tile; k++) {
                accum_packed = op::template op<RT2>(accum_packed, src.tiles[i][j].data[k]);
            }
        }

        RT accum_single = op::template op<RT>(accum_packed.x, accum_packed.y);

        // Now we need to do a lil shuffle to make everyone happy.

        accum_single = op::template op<RT>(accum_single, packed_shfl_down(MASK_ALL, accum_single, 32));
        accum_single = op::template op<RT>(accum_single, packed_shfl_down(MASK_ALL, accum_single, 16));

        if(reset) {
            col_accum[j][0] = accum_single;
        }
        else {
            col_accum[j][0] = op::template op<RT>(src_accum[j][0], accum_single);
        }

        col_accum[j][0] = packed_shfl(MASK_ALL, col_accum[j][0], leader);
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// two-operand row reductions. (Accumulate and REPLACE.)
/**
 * @brief Store the maximum of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__attribute__((device)) static inline void row_max(V &row_accum, const T &src) {
    row_reduce<base_ops::max, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the minimum of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__attribute__((device)) static inline void row_min(V &row_accum, const T &src) {
    row_reduce<base_ops::min, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the sum of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__attribute__((device)) static inline void row_sum(V &row_accum, const T &src) {
    row_reduce<base_ops::sum, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the product of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__attribute__((device)) static inline void row_prod(V &row_accum, const T &src) {
    row_reduce<base_ops::mul, V, T, true>(row_accum, src, row_accum);
}
// three-operand row reductions. (Accumulate ONTO.)
/**
 * @brief Store the maximum of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__attribute__((device)) static inline void row_max(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::max, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the minimum of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__attribute__((device)) static inline void row_min(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::min, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the sum of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__attribute__((device)) static inline void row_sum(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::sum, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the product of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__attribute__((device)) static inline void row_prod(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::mul, V, T, false>(row_accum, src, src_accum);
}

// two-operand col reductions. (Accumulate and REPLACE.)

/**
 * @brief Store the maximum of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__attribute__((device)) static inline void col_max(V &col_accum, const T &src) {
    col_reduce<base_ops::max, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the minimum of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__attribute__((device)) static inline void col_min(V &col_accum, const T &src) {
    col_reduce<base_ops::min, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the sum of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__attribute__((device)) static inline void col_sum(V &col_accum, const T &src) {
    col_reduce<base_ops::sum, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the product of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__attribute__((device)) static inline void col_prod(V &col_accum, const T &src) {
    col_reduce<base_ops::mul, V, T, true>(col_accum, src, col_accum);
}
// three-operand col reductions. (Accumulate ONTO.)
/**
 * @brief Store the maximum of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__attribute__((device)) static inline void col_max(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::max, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the minimum of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__attribute__((device)) static inline void col_min(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::min, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the sum of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__attribute__((device)) static inline void col_sum(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::sum, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the product of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__attribute__((device)) static inline void col_prod(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::mul, V, T, false>(col_accum, src, src_accum);
}

}
# 11 "/root/HipKittens//include/ops/warp/register/tile/tile.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/register/tile/mma.cuh" 1
/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in registers.
 */






namespace kittens {


__attribute__((device)) static inline void mfma161616(float2 (&D)[2],
                                        const half_2 (&A)[2],
                                        const half_2 (&B)[2],
                                        const float2 (&C)[2]) {
    typedef __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16 half4_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float float4_t;
    *(float4_t*)D = __builtin_amdgcn_mfma_f32_16x16x16f16(
        *(half4_t*)A,
        *(half4_t*)B,
        *(float4_t*)C,
        0, 0, 0
    );
}

__attribute__((device)) static inline void mfma161616(float2 (&D)[2],
                                        const bf16_2 (&A)[2],
                                        const bf16_2 (&B)[2],
                                        const float2 (&C)[2]) {
    typedef __attribute__((__vector_size__(4 * sizeof(short)))) short short4_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float float4_t;
    *(float4_t*)D = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
        *(short4_t*)A,
        *(short4_t*)B,
        *(float4_t*)C,
        0, 0, 0
    );
}


/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__attribute__((device)) static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col> &d,
                                    const rt_base<half, ducks::rt_layout::row> &a,
                                    const rt_base<half, ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__attribute__((device)) static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col> &d,
                                    const rt_base<bf16, ducks::rt_layout::row> &a,
                                    const rt_base<bf16, ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
/**
 * @brief Base dot product operation for row layout.
 *
 * This function performs the base dot product operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__attribute__((device)) static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<half, ducks::rt_layout::row> &a,
                                     const rt_base<half, ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__attribute__((device)) static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<bf16, ducks::rt_layout::row> &a,
                                     const rt_base<bf16, ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__attribute__((device)) static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<half, ducks::rt_layout::col> &a,
                                     const rt_base<half, ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__attribute__((device)) static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<bf16, ducks::rt_layout::col> &a,
                                     const rt_base<bf16, ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A and B.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__attribute__((device)) static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::col> &d,
                                      const rt_base<half, ducks::rt_layout::col> &a,
                                      const rt_base<half, ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__attribute__((device)) static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::col> &d,
                                      const rt_base<bf16, ducks::rt_layout::col> &a,
                                      const rt_base<bf16, ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
/**
 * @brief Matrix multiply-accumulate operation.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` function.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_hf<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_hf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_hf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_hf<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::col_layout B, ducks::rt::col_layout C>
__attribute__((device)) static inline void mma_AB(D &d,
                               const A &a,
                               const B &b,
                               const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::cols); // Check D matches A, B
    static_assert(A::cols == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

#pragma unroll
    for(int n = 0; n < D::height; n++) {
#pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AB_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[0][m],
                c.tiles[n][m]
            );
#pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_AB_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}

/**
 * @brief Dot product operation for row layout.
 *
 * This function performs the dot product operation
 * using the `hmma16816` function.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
__attribute__((device)) static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b, // notice row and (M, K) instead of col and (K, M)
                                const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    constexpr auto _ = D::height; 
    constexpr auto __ = D::width; 

#pragma unroll
    for(int n = 0; n < D::height; n++) {
#pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_ABt_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[m][0],
                c.tiles[n][m]
            );
#pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_ABt_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}
/**
 * @brief Matrix multiply-accumulate operation with transposed A.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, row_layout> matrix.
 * @param[in] b The second input rt_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::col_layout D, ducks::rt::col_layout A, ducks::rt::col_layout B, ducks::rt::col_layout C>
__attribute__((device)) static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {
    static_assert(D::rows == A::cols && D::cols == B::cols); // Check D matches A, B
    static_assert(A::rows == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

#pragma unroll
    for(int n = 0; n < D::height; n++) {
#pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtB_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[0][m],
                c.tiles[n][m]
            );
#pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtB_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}

/**
 * @brief Matrix multiply-accumulate operation with transposed A and B.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, col_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::col_layout D, ducks::rt::col_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
__attribute__((device)) static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b,
                                 const C &c) {
    static_assert(D::rows == A::cols && D::cols == B::rows); // Check D matches A, B
    static_assert(A::rows == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

#pragma unroll
    for(int n = 0; n < D::height; n++) {
#pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtBt_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[m][0],
                c.tiles[n][m]
            );
#pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtBt_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}
}
# 12 "/root/HipKittens//include/ops/warp/register/tile/tile.cuh" 2
# 9 "/root/HipKittens//include/ops/warp/register/register.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/register/vec/vec.cuh" 1
/**
 * @file
 * @brief An aggregate header for warp operations on register vectors.
 */



# 1 "/root/HipKittens//include/ops/warp/register/vec/conversions.cuh" 1
/**
 * @file
 * @brief Conversions on vectors stored in registers.
 */






namespace kittens {

template<ducks::rv::all RV2, ducks::rv::all RV1>
__attribute__((device)) static inline void copy(RV2 &dst, const RV1 &src) {
    static_assert(RV1::length == RV2::length, "Register vectors must be the same length.");
    using D1 = RV1::dtype;
    using D2 = RV2::dtype;

    using D1_1 = base_types::packing<D1>::unpacked_type;
    using D1_2 = base_types::packing<D1_1>::packed_type;

    using D2_1 = base_types::packing<D2>::unpacked_type;
    using D2_2 = base_types::packing<D2_1>::packed_type;

    if constexpr (std::is_same_v<typename RV1::layout, typename RV2::layout>) { // just a simple copy /typecast
#pragma unroll
        for(int i = 0; i < RV1::outer_dim; i++) {
#pragma unroll
            for(int j = 0; j < RV1::inner_dim; j++) {
                dst[i][j] = base_types::convertor<D2, D1>::convert(src[i][j]);
            }
        }
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::naive> && std::is_same_v<typename RV2::layout, ducks::rv_layout::align>) {
        static_assert(false, "Unsupported layout conversion");
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::align> && std::is_same_v<typename RV2::layout, ducks::rv_layout::naive>) {
        static_assert(false, "Unsupported layout conversion");
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::align> && std::is_same_v<typename RV2::layout, ducks::rv_layout::ortho>) {
        static_assert(false, "Unsupported layout conversion");
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::ortho> && std::is_same_v<typename RV2::layout, ducks::rv_layout::align>) {
        static_assert(false, "Unsupported layout conversion");
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::ortho> && std::is_same_v<typename RV2::layout, ducks::rv_layout::naive>) {
        static_assert(false, "Unsupported layout conversion");
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::naive> && std::is_same_v<typename RV2::layout, ducks::rv_layout::ortho>) {
        static_assert(false, "Unsupported layout conversion");
    } else {
        static_assert(false, "Unsupported layout conversion");
    }
}

} // namespace kittens
# 9 "/root/HipKittens//include/ops/warp/register/vec/vec.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/register/vec/maps.cuh" 1
/**
 * @file
 * @brief Maps on vectors stored in registers.
 */






namespace kittens {

/* ----------  Vector Maps  ---------- */

/**
 * @brief Perform a unary operation on a vector.
 *
 * @tparam op The unary operation to perform.
 * @tparam T The type of the vector.
 * @param dst[out] The destination vector where the result is stored.
 * @param src[in] The source vector to perform the operation on.
 */
template<typename op, ducks::rv::all T>
__attribute__((device)) static inline void unary_op(T &dst, const T &src) {
#pragma unroll
    for(int i = 0; i < dst.outer_dim; i++) {
#pragma unroll
        for(int j = 0; j < dst.inner_dim; j++) {
            dst[i][j] = op::template op<typename T::dtype>(src[i][j]);
        }
    }
}


/**
 * @brief Apply a unary op to a flattened subrange [start, end) of a 2D vector.
 */
 template<typename op, ducks::rv::all T>
 __attribute__((device)) static inline void unary_op_range(T &dst, const T &src, int start, int end) {
     const int O = dst.outer_dim;
     const int I = dst.inner_dim;
     // flat index over O*I; map back to (i,j)
#pragma unroll
     for (int n = start; n < end; ++n) {
         const int i = n / I;
         const int j = n - i * I;
         dst[i][j] = op::template op<typename T::dtype>(src[i][j]);
     }
 }

 /** first half */
 template<typename op, ducks::rv::all T>
 __attribute__((device)) static inline void unary_op_first_half_vec(T &dst, const T &src) {
     const int N = dst.outer_dim * dst.inner_dim;
     unary_op_range<op, T>(dst, src, 0, N >> 1);
 }

 /** second half */
 template<typename op, ducks::rv::all T>
 __attribute__((device)) static inline void unary_op_second_half_vec(T &dst, const T &src) {
     const int N = dst.outer_dim * dst.inner_dim;
     unary_op_range<op, T>(dst, src, N >> 1, N);
 }

 // --- exp shortcuts ---
 template<ducks::rv::all T>
 __attribute__((device)) static inline void exp_first_half_vec(T &dst, const T &src) {
     unary_op_first_half_vec<base_ops::exp2, T>(dst, src);
 }
 template<ducks::rv::all T>
 __attribute__((device)) static inline void exp_second_half_vec(T &dst, const T &src) {
     unary_op_second_half_vec<base_ops::exp2, T>(dst, src);
 }





/**
 * @brief Perform a binary operation on two vectors.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vectors.
 * @param dst[out] The destination vector where the result is stored.
 * @param lhs[in] The left-hand side vector for the operation.
 * @param rhs[in] The right-hand side vector for the operation.
 */
template<typename op, ducks::rv::all T>
__attribute__((device)) static inline void bin_op(T &dst, const T &lhs, const T &rhs) {
#pragma unroll
    for(int i = 0; i < dst.outer_dim; i++) {
#pragma unroll
        for(int j = 0; j < dst.inner_dim; j++) {
            dst[i][j] = op::template op<typename T::dtype>(lhs[i][j], rhs[i][j]);
        }
    }
}
/**
 * @brief Perform a binary operation on a vector and a scalar.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vector.
 * @param dst[out] The destination vector where the result is stored.
 * @param src[in] The source vector for the operation.
 * @param param[in] The scalar parameter for the operation.
 */
template<typename op, ducks::rv::all T>
__attribute__((device)) static inline void bin_op(T &dst, const T &src, const typename T::dtype &param) {
#pragma unroll
    for(int i = 0; i < dst.outer_dim; i++) {
#pragma unroll
        for(int j = 0; j < dst.inner_dim; j++) {
            dst[i][j] = op::template op<typename T::dtype>(src[i][j], param);
        }
    }
}
/**
 * @brief Perform a binary operation on a vector and an unpacked scalar.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vector.
 * @param dst[out] The destination vector where the result is stored.
 * @param src[in] The source vector for the operation.
 * @param param[in] The unpacked scalar parameter for the operation.
 */
template<typename op, ducks::rv::tile_layout T>
requires (!std::is_same_v<typename T::dtype, typename base_types::packing<typename T::dtype>::unpacked_type>)
__attribute__((device)) static inline void bin_op(T &dst, const T &src, const typename base_types::packing<typename T::dtype>::unpacked_type &param) {
    bin_op<op, T>(dst, src, base_types::packing<typename T::dtype>::pack(param));
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// ---- const ops ----

/**
 * @brief Sets all elements of a register vector to zero.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector to be set to zero.
 */
template<ducks::rv::all T>
__attribute__((device)) static inline void zero(T &dst) {
    unary_op<base_ops::zero, T>(dst, dst);
}
/**
 * @brief Sets all elements of a register vector to one.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector to be set to one.
 */
template<ducks::rv::all T>
__attribute__((device)) static inline void one(T &dst) {
    unary_op<base_ops::one, T>(dst, dst);
}
/**
 * @brief Sets all elements of a register vector to positive infinity.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector to be set to positive infinity.
 */
template<ducks::rv::all T>
__attribute__((device)) static inline void pos_infty(T &dst) {
    unary_op<base_ops::pos_infty, T>(dst, dst);
}
/**
 * @brief Sets all elements of a register vector to negative infinity.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector to be set to negative infinity.
 */
template<ducks::rv::all T>
__attribute__((device)) static inline void neg_infty(T &dst) {
    unary_op<base_ops::neg_infty, T>(dst, dst);
}

// ---- unary ops ----

/**
 * @brief Copies the elements from one register vector to another.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the source vector.
 * @param dst[out] Destination vector where the elements will be copied to.
 * @param src[in] Source vector to copy the elements from.
 */
template<ducks::rv::all T, typename U>
__attribute__((device)) static inline void copy(T &dst, const U &src) {
    bin_op<base_ops::copy2, T>(dst, dst, src); // the second arg is ignored here.
}
/**
 * @brief Applies the exponential function element-wise to a register vector.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template<ducks::rv::all T>
__attribute__((device)) static inline void exp(T &dst, const T &src) {
    unary_op<base_ops::exp, T>(dst, src);
}
/**
 * @brief Applies the exponential function element-wise to a register vector, in base 2.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template<ducks::rv::all T>
__attribute__((device)) static inline void exp2(T &dst, const T &src) {
    unary_op<base_ops::exp2, T>(dst, src);
}
/**
 * @brief Applies the natural logarithm function element-wise to a register vector.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template<ducks::rv::all T>
__attribute__((device)) static inline void log(T &dst, const T &src) {
    unary_op<base_ops::log, T>(dst, src);
}
/**
 * @brief Applies the logarithm base 2 function element-wise to a register vector.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the logarithm base 2 function to.
 */
template<ducks::rv::all T>
__attribute__((device)) static inline void log2(T &dst, const T &src) {
    unary_op<base_ops::log2, T>(dst, src);
}
/**
 * @brief Applies the absolute value function element-wise to a register vector.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector where the absolute values will be stored.
 * @param src[in] Source vector to apply the absolute value function to.
 */
template<ducks::rv::all T>
__attribute__((device)) static inline void abs(T &dst, const T &src) {
    unary_op<base_ops::abs, T>(dst, src);
}
/**
 * @brief Applies the rectified linear unit (ReLU) function element-wise to a register vector.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector where the ReLU values will be stored.
 * @param src[in] Source vector to apply the ReLU function to.
 */
template<ducks::rv::all T>
__attribute__((device)) static inline void relu(T &dst, const T &src) {
    unary_op<base_ops::relu, T>(dst, src);
}

// ---- binary ops ----

/**
 * @brief Computes the element-wise maximum of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the maximum values will be stored.
 * @param lhs[in] First vector for the maximum operation.
 * @param rhs[in] Second vector for the maximum operation.
 */
template<ducks::rv::all T, typename U>
__attribute__((device)) static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::max, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise minimum of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the minimum values will be stored.
 * @param lhs[in] First vector for the minimum operation.
 * @param rhs[in] Second vector for the minimum operation.
 */
template<ducks::rv::all T, typename U>
__attribute__((device)) static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::min, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise sum of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the sum values will be stored.
 * @param lhs[in] First vector for the sum operation.
 * @param rhs[in] Second vector for the sum operation.
 */
template<ducks::rv::all T, typename U>
__attribute__((device)) static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sum, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise difference of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the difference values will be stored.
 * @param lhs[in] First vector for the difference operation.
 * @param rhs[in] Second vector for the difference operation.
 */
template<ducks::rv::all T, typename U>
__attribute__((device)) static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sub, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise product of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the product values will be stored.
 * @param lhs[in] First vector for the product operation.
 * @param rhs[in] Second vector for the product operation.
 */
template<ducks::rv::all T, typename U>
__attribute__((device)) static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::mul, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise division of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the division values will be stored.
 * @param lhs[in] First vector for the division operation.
 * @param rhs[in] Second vector for the division operation.
 */
template<ducks::rv::all T, typename U>
__attribute__((device)) static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::div, T>(dst, lhs, rhs);
}

}
# 10 "/root/HipKittens//include/ops/warp/register/vec/vec.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/register/vec/reductions.cuh" 1
/**
 * @file
 * @brief Reductions on vectors stored in registers.
 */






namespace kittens {

/* ----------  Vector Reductions  ---------- */

/**
 * @brief Performs a reduction operation on elements of a register vector within a warp.
 *
 * This function applies a specified operation to reduce the elements of a register vector `src` to a single value.
 * The result is stored in `accum`. If the `reset` parameter is true, the reduction includes an initial value `src_accum`.
 * The reduction operation is performed in a warp-wide context, ensuring synchronization between threads in the warp.
 *
 * @tparam op The operation to perform on the elements. Must provide a static `op` method.
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @tparam reset A boolean flag indicating whether to include an initial value in the reduction.
 * @param[out] accum The result of the reduction operation.
 * @param[in] src The register vector to reduce.
 * @param[in] src_accum The initial value to include in the reduction if `reset` is false.
 */
template<typename op, ducks::rv::all RV, bool reset>
__attribute__((device)) static inline void reduce(
        typename base_types::packing<typename RV::dtype>::unpacked_type &dst_accum,
        const RV &src,
        const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    using T = base_types::packing<typename RV::dtype>::unpacked_type;
    int laneid = kittens::laneid();
    if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        static_assert(false, "ortho_l reduce is not currently supported");
    }
    else if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        static_assert(false, "align_l reduce is not currently supported");
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        T accum = src[0][0];
#pragma unroll
        for(int i = 1; i < src.outer_dim; i++) {
            if (i < src.outer_dim-1 || i*64 + laneid < src.length) { // Changed from TILE_ROW_DIM<T>*2 to 64
                accum = op::template op<T>(accum, src[i][0]);
            }
        }

        // Reduce across all 64 lanes
        if (src.length > 32) accum = op::template op<T>(accum, packed_shfl_down(kittens::MASK_ALL, accum, 32));
        if (src.length > 16) accum = op::template op<T>(accum, packed_shfl_down(kittens::MASK_ALL, accum, 16));
        if (src.length > 8) accum = op::template op<T>(accum, packed_shfl_down(kittens::MASK_ALL, accum, 8));
        if (src.length > 4) accum = op::template op<T>(accum, packed_shfl_down(kittens::MASK_ALL, accum, 4));
        if (src.length > 2) accum = op::template op<T>(accum, packed_shfl_down(kittens::MASK_ALL, accum, 2));
        if (src.length > 1) accum = op::template op<T>(accum, packed_shfl_down(kittens::MASK_ALL, accum, 1));

        if constexpr (!reset) accum = op::template op<T>(accum, src_accum);
        dst_accum = packed_shfl(kittens::MASK_ALL, accum, 0);
    }
}


/**
 * @brief Finds the maximum element in a register vector.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] max_val The maximum value found in the vector.
 * @param[in] src The register vector to find the maximum in.
 */
template<ducks::rv::all RV>
__attribute__((device)) static inline void max(typename base_types::packing<typename RV::dtype>::unpacked_type &max_val, const RV &src) {
    reduce<base_ops::max, RV, true>(max_val, src, max_val);
}

/**
 * @brief Finds the minimum element in a register vector.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] min_val The minimum value found in the vector.
 * @param[in] src The register vector to find the minimum in.
 */
template<ducks::rv::all RV>
__attribute__((device)) static inline void min(typename base_types::packing<typename RV::dtype>::unpacked_type &min_val, const RV &src) {
    reduce<base_ops::min, RV, true>(min_val, src, min_val);
}

/**
 * @brief Calculates the sum of elements in a register vector.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] sum_val The sum of the values in the vector.
 * @param[in] src The register vector to sum.
 */
template<ducks::rv::all RV>
__attribute__((device)) static inline void sum(typename base_types::packing<typename RV::dtype>::unpacked_type &sum_val, const RV &src) {
    reduce<base_ops::sum, RV, true>(sum_val, src, sum_val);
}

/**
 * @brief Calculates the product of elements in a register vector.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] prod_val The product of the values in the vector.
 * @param[in] src The register vector to multiply.
 */
template<ducks::rv::all RV>
__attribute__((device)) static inline void prod(typename base_types::packing<typename RV::dtype>::unpacked_type &prod_val, const RV &src) {
    reduce<base_ops::mul, RV, true>(prod_val, src, prod_val);
}

// Three operand versions.

/**
 * @brief Finds the maximum element in a register vector and accumulates it with src_accum.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] max_val The maximum value found in the vector, accumulated with src_accum.
 * @param[in] src The register vector to find the maximum in.
 * @param[in] src_accum The initial value to accumulate with the maximum value found.
 */
template<ducks::rv::all RV>
__attribute__((device)) static inline void max(typename base_types::packing<typename RV::dtype>::unpacked_type &max_val, const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    reduce<base_ops::max, RV, false>(max_val, src, src_accum);
}

/**
 * @brief Finds the minimum element in a register vector and accumulates it with src_accum.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] min_val The minimum value found in the vector, accumulated with src_accum.
 * @param[in] src The register vector to find the minimum in.
 * @param[in] src_accum The initial value to accumulate with the minimum value found.
 */
template<ducks::rv::all RV>
__attribute__((device)) static inline void min(typename base_types::packing<typename RV::dtype>::unpacked_type &min_val, const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    reduce<base_ops::min, RV, false>(min_val, src, src_accum);
}

/**
 * @brief Calculates the sum of elements in a register vector and accumulates it with src_accum.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] sum_val The sum of the values in the vector, accumulated with src_accum.
 * @param[in] src The register vector to sum.
 * @param[in] src_accum The initial value to accumulate with the sum of the vector.
 */
template<ducks::rv::all RV>
__attribute__((device)) static inline void sum(typename base_types::packing<typename RV::dtype>::unpacked_type &sum_val, const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    reduce<base_ops::sum, RV, false>(sum_val, src, src_accum);
}

/**
 * @brief Calculates the product of elements in a register vector and accumulates it with src_accum.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] prod_val The product of the values in the vector, accumulated with src_accum.
 * @param[in] src The register vector to multiply.
 * @param[in] src_accum The initial value to accumulate with the product of the vector.
 */
template<ducks::rv::all RV>
__attribute__((device)) static inline void prod(typename base_types::packing<typename RV::dtype>::unpacked_type &prod_val, const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    reduce<base_ops::mul, RV, false>(prod_val, src, src_accum);
}

}
# 10 "/root/HipKittens//include/ops/warp/register/vec/vec.cuh" 2
# 9 "/root/HipKittens//include/ops/warp/register/register.cuh" 2
# 12 "/root/HipKittens//include/ops/warp/warp.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/shared/shared.cuh" 1
/**
 * @file
 * @brief An aggregate header of warp operations on data in shared memory
 */



# 1 "/root/HipKittens//include/ops/warp/shared/tile/tile.cuh" 1
/**
 * @file
 * @brief An aggregate header for warp operations on shared tiles.
 */



# 1 "/root/HipKittens//include/ops/warp/shared/tile/conversions.cuh" 1
/**
 * @file
 * @brief Conversions between shared tile types.
 */






namespace kittens {

/* ----------  COPIES  ---------- */

/**
 * @brief Copies data from one shared memory tile to another, potentially with different data types and layouts.
 *
 * @tparam T The data type of the destination tile.
 * @tparam U The data type of the source tile.
 * @tparam _height The height of the tile.
 * @tparam _width The width of the tile.
 * @tparam L1 The layout of the destination tile.
 * @tparam L2 The layout of the source tile.
 * @param[out] dst The destination tile.
 * @param[in] src The source tile.
 */
template<typename T, typename U, int _height, int _width>
__attribute__((device)) static inline void copy(st<T, _height, _width> &dst, const st<U, _height, _width> &src) {
#pragma unroll
    for(int i = laneid(); i < dst.num_elements; i+=kittens::WARP_THREADS) {
        int row = i/dst.cols, col = i%dst.cols;
        dst[{row, col}] = base_types::convertor<T, U>::convert(src[{row, col}]);
    }
}

/* ----------  SUBTILE  ---------- */

/**
* @brief Returns a reference to a subtile of the given shared tile.
*
* @tparam subtile_height The height of the subtile.
* @tparam subtile_width The width of the subtile.
* @tparam ST The type of the input tile, which must satisfy the ducks::st::all concept.
* @param src The input tile.
* @param row_idx The row coord of the subtile, in units of subtile_height*16 elements.
* @param col_idx The col coord of the subtile, in units of subtile_width*16 elements.
* @return A reference to the subtile.
*
* @note The subtile {height, width} must evenly divide the tile {height, width}.
*/
template<int subtile_rows, int subtile_cols, ducks::st::all ST>
__attribute__((device)) inline st_subtile<ST, subtile_rows, subtile_cols> subtile_inplace(ST &src, int2 rowcol, bool unformatted = false) {
    using T = typename ST::dtype;
    static_assert(subtile_rows % TILE_ROW_DIM<T> == 0);
    static_assert(subtile_cols % TILE_COL_DIM<T> == 0);
    static_assert(ST::height % (subtile_rows/TILE_ROW_DIM<T>) == 0);
    static_assert(ST::width % (subtile_cols/TILE_COL_DIM<T>) == 0);
    static_assert(ST::height == ST::underlying_height && ST::width == ST::underlying_width); // must be a real ST, no recursive subtiles.
    return st_subtile<ST, subtile_rows, subtile_cols>(src, rowcol);
}

} // namespace kittens
# 9 "/root/HipKittens//include/ops/warp/shared/tile/tile.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/shared/tile/maps.cuh" 1
/**
 * @file
 * @brief Warp-scope maps on shared tiles.
 */






namespace kittens {

/* ----------  Uniform tile maps (independent of layout)  ---------- */

/**
 * @brief Performs a uniform unary operation on a tile.
 * 
 * This function applies a given unary operation to each element of the source tile and stores the result in the destination tile.
 * The operation is applied independently to each element, without considering its position or the values of neighboring elements.
 * 
 * @tparam op The unary operation to be applied. Must be specialized to support operation on the data type of T.
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the unary operation is applied.
 */
template<typename op, ducks::st::all T> // T2, w, h can be inferred from dst as long as op is specialized
__attribute__((device)) static inline void unary_map(T &dst, const T &src) {
#pragma unroll
    for(int i = kittens::laneid(); i < dst.num_elements; i += WARP_THREADS) {
        dst.data[i] = op::template op<typename T::dtype>(src.data[i]);
    }
}

/**
 * @brief Performs a uniform binary operation on a tile with a scalar parameter.
 * 
 * This function applies a given binary operation to each element of the source tile and a scalar parameter, then stores the result in the destination tile.
 * The operation is applied independently to each element, treating the scalar parameter as the second operand for each operation.
 * 
 * @tparam op The binary operation to be applied. Must be specialized to support operation on the data type of T and the scalar parameter.
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the binary operation is applied.
 * @param[in] param The scalar parameter to be used as the second operand in the binary operation.
 */
template<typename op, ducks::st::all T>
__attribute__((device)) static inline void bin_map(T &dst, const T &src, const typename T::dtype &param) {
#pragma unroll
    for(int i = kittens::laneid(); i < dst.num_elements; i += WARP_THREADS) {
        dst.data[i] = op::template op<typename T::dtype>(src.data[i], param);
    }
}

/**
 * @brief Performs a uniform binary operation on two tiles.
 * 
 * This function applies a given binary operation to corresponding elements of two source tiles and stores the result in the destination tile.
 * The operation is applied independently to each pair of elements, without considering their positions or the values of neighboring elements.
 * 
 * @tparam op The binary operation to be applied. Must be specialized to support operation on the data type of T.
 * @tparam T The type of the tiles. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile to which the binary operation is applied.
 * @param[in] rhs The second source tile to which the binary operation is applied.
 */
template<typename op, ducks::st::all T>
__attribute__((device)) static inline void bin_map(T &dst, const T &lhs, const T &rhs) {
#pragma unroll
    for(int i = kittens::laneid(); i < dst.num_elements; i += WARP_THREADS) {
        int row = i/dst.cols, col = i%dst.cols;
        dst.data[i] = op::template op<typename T::dtype>(lhs.data[i], rhs.data[i]);
    }
}

/**
 * @brief Performs a row-wise binary operation on a tile with a vector.
 * 
 * This function applies a given binary operation to each row of the source tile and the corresponding element of the source vector,
 * then stores the result in the destination tile. The operation is applied independently to each row, using the vector element as 
 * the second operand for each element in the row.
 * 
 * @tparam op The binary operation to be applied. Must be specialized to support operation on the data type of T and the vector elements.
 * @tparam T The type of the tiles. Must satisfy the `ducks::st::all` concept.
 * @tparam V The type of the vector. Must have the same data type as T.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the binary operation is applied.
 * @param[in] vec The source vector containing the second operand for each row operation.
 */
template<typename op, ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void row_map(T &dst, const T &src, const V &vec) {
    static_assert(std::is_same<typename T::dtype, typename V::dtype>::value, "Tile and vector must have the same data type");
    static_assert(V::length == T::rows, "Vector length must match the number of rows in the tile");
#pragma unroll
    for(int i = kittens::laneid(); i < dst.num_elements; i += WARP_THREADS) {
        int row = i/dst.cols, col = i%dst.cols;
        dst[{row, col}] = op::template op<typename T::dtype>(src[{row, col}], vec[row]);
    }
}

/**
 * @brief Performs a column-wise binary operation on a tile with a vector.
 * 
 * This function applies a given binary operation to each column of the source tile and the corresponding element of the source vector,
 * then stores the result in the destination tile. The operation is applied independently to each column, using the vector element as 
 * the second operand for each element in the column.
 * 
 * @tparam op The binary operation to be applied. Must be specialized to support operation on the data type of T and the vector elements.
 * @tparam T The type of the tiles. Must satisfy the `ducks::st::all` concept.
 * @tparam V The type of the vector. Must have the same data type as T.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the binary operation is applied.
 * @param[in] vec The source vector containing the second operand for each column operation.
 */
template<typename op, ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void col_map(T &dst, const T &src, const V &vec) {
    static_assert(std::is_same<typename T::dtype, typename V::dtype>::value, "Tile and vector must have the same data type");
    static_assert(V::length == T::cols, "Vector length must match the number of columns in the tile");
#pragma unroll
    for(int i = kittens::laneid(); i < dst.num_elements; i += WARP_THREADS) {
        int row = i/dst.cols, col = i%dst.cols;
        dst[{row, col}] = op::template op<typename T::dtype>(src[{row, col}], vec[col]);
    }
}


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// All of the annoying qualifiers *should* be automatically inferred during compile-time.
// So, syntax should just be kittens::add_row(tile, colvec);

// const maps
/**
 * @brief Sets all elements of the destination tile to zero.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void zero(T &dst) {
    unary_map<base_ops::zero, T>(dst, dst);
}
/**
 * @brief Sets all elements of the destination tile to one.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void one(T &dst) {
    unary_map<base_ops::one, T>(dst, dst);
}
/**
 * @brief Sets all elements of the destination tile to positive infinity.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void pos_infty(T &dst) {
    unary_map<base_ops::pos_infty, T>(dst, dst);
}
/**
 * @brief Sets all elements of the destination tile to negative infinity.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void neg_infty(T &dst) {
    unary_map<base_ops::neg_infty, T>(dst, dst);
}

// unary maps
/**
 * @brief Applies the exponential function to each element of the source tile and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the exponential function is applied.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void exp(T &dst, const T &src) {
    unary_map<base_ops::exp, T>(dst, src);
}
/**
 * @brief Applies the exponential function to each element of the source tile and stores the result in the destination tile, in base 2.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the exponential function is applied.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void exp2(T &dst, const T &src) {
    unary_map<base_ops::exp2, T>(dst, src);
}
/**
 * @brief Applies the natural logarithm function to each element of the source tile and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the natural logarithm function is applied.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void log(T &dst, const T &src) {
    unary_map<base_ops::log, T>(dst, src);
}
/**
 * @brief Applies the logarithm base 2 function to each element of the source tile and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the logarithm base 2 function is applied.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void log2(T &dst, const T &src) {
    unary_map<base_ops::log2, T>(dst, src);
}
/**
 * @brief Applies the absolute function to each element of the source tile and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the absolute function is applied.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void abs(T &dst, const T &src) {
    unary_map<base_ops::abs, T>(dst, src);
}
/**
 * @brief Applies the rectified linear unit function to each element of the source tile and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the rectified linear unit function is applied.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void relu(T &dst, const T &src) {
    unary_map<base_ops::relu, T>(dst, src);
}
/**
 * @brief Copies the elements of the source tile to the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source data to be copied.
 */
template<ducks::st::all T, typename U>
__attribute__((device)) static inline void copy(T &dst, const U &src) {
    bin_map<base_ops::copy, T>(dst, src);
}

// uniform binary maps
/**
 * @brief Finds the maximum of each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__attribute__((device)) static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::max, T>(dst, lhs, rhs);
}
/**
 * @brief Finds the minimum of each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__attribute__((device)) static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::min, T>(dst, lhs, rhs);
}
/**
 * @brief Adds each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__attribute__((device)) static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sum, T>(dst, lhs, rhs);
}
/**
 * @brief Subtracts each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__attribute__((device)) static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sub, T>(dst, lhs, rhs);
}
/**
 * @brief Multiplies each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__attribute__((device)) static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::mul, T>(dst, lhs, rhs);
}
/**
 * @brief Divides each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__attribute__((device)) static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::div, T>(dst, lhs, rhs);
}

// Row and col maps

/**
 * @brief Adds row values to each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the addition on.
 * @param row_values[in] Column vector containing values to add to each row.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void add_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sum, T, V>(dst, src, row_values);
}
/**
 * @brief Subtracts row values from each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the subtraction on.
 * @param row_values[in] Column vector containing values to subtract from each row.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void sub_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sub, T, V>(dst, src, row_values);
}
/**
 * @brief Multiplies each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the multiplication on.
 * @param row_values[in] Column vector containing values to multiply each row by.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void mul_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::mul, T, V>(dst, src, row_values);
}
/**
 * @brief Divides each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the division on.
 * @param row_values[in] Column vector containing values to divide each row by.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void div_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::div, T, V>(dst, src, row_values);
}
/**
 * @brief Broadcast a vector into into a tile's rows.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param row_values[in] Column vector containing values to broadcast into rows.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void broadcast_row(T &dst, const V &row_values) {
    row_map<base_ops::copy2, T, V>(dst, dst, row_values);
}


// col maps
/**
 * @brief Adds column values to each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the addition on.
 * @param col_values[in] Row vector containing values to add to each column.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void add_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sum, T, V>(dst, src, col_values);
}
/**
 * @brief Subtracts column values from each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the subtraction on.
 * @param col_values[in] Row vector containing values to subtract from each column.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void sub_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sub, T, V>(dst, src, col_values);
}
/**
 * @brief Multiplies each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the multiplication on.
 * @param col_values[in] Row vector containing values to multiply each column by.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void mul_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::mul, T, V>(dst, src, col_values);
}
/**
 * @brief Divides each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the division on.
 * @param col_values[in] Row vector containing values to divide each column by.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void div_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::div, T, V>(dst, src, col_values);
}
/**
 * @brief Broadcast a vector into into a tile's columns.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param row_values[in] Row vector containing values to broadcast into cols.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void broadcast_col(T &dst, const V &col_values) {
    col_map<base_ops::copy2, T, V>(dst, dst, col_values);
}

}
# 10 "/root/HipKittens//include/ops/warp/shared/tile/tile.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/shared/tile/reductions.cuh" 1
/**
 * @file
 * @brief Warp-scope reductions on shared tiles.
 */






namespace kittens {

/**
 * Performs row-wise reduction on a matrix using a specified operation.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type with row layout.
 * @param row_accum The accumulator where the result of the reduction is stored.
 * @param src The source matrix on which to perform the reduction.
 * @param src_accum The initial value of the accumulator, used when reset is false.
 * @param reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 */
template<typename op, ducks::sv::all V, ducks::st::all T, bool reset>
__attribute__((device)) static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    using dtype = typename V::dtype;
    for (int row = kittens::laneid(); row < src.rows; row += kittens::WARP_THREADS) {
        dtype accum = src[{row, 0}];
#pragma unroll
        for (int col = 1; col < src.cols; col++) {
            accum = op::template op<dtype>(accum, src[{row, col}]);
        }
        if (reset) {
            row_accum[row] = accum;
        } else {
            row_accum[row] = op::template op<dtype>(src_accum[row], accum);
        }
    }
    __syncthreads();
}

/**
 * Performs column-wise reduction on a matrix using a specified operation.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The shared vector type for the column accumulator.
 * @tparam T The shared matrix type with column layout.
 * @param col_accum The accumulator where the result of the reduction is stored.
 * @param src The source matrix on which to perform the reduction.
 * @param src_accum The initial value of the accumulator, used when reset is false.
 * @param reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 */
template<typename op, ducks::sv::all V, ducks::st::all T, bool reset>
__attribute__((device)) static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    using dtype = typename V::dtype;
    for (int col = kittens::laneid(); col < src.cols; col += kittens::WARP_THREADS) {
        dtype accum = src[{0, col}];
#pragma unroll
        for (int row = 1; row < src.rows; row++) {
            accum = op::template op<dtype>(accum, src[{row, col}]);
        }
        if (reset) {
            col_accum[col] = accum;
        } else {
            col_accum[col] = op::template op<dtype>(src_accum[col], accum);
        }
    }
    __syncthreads();
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

/**
 * @brief Store the maximum of each row of the src shared matrix in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void row_max(V &row_accum, const T &src) {
    row_reduce<base_ops::max, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the minimum of each row of the src shared matrix in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void row_min(V &row_accum, const T &src) {
    row_reduce<base_ops::min, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the sum of each row of the src shared matrix in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void row_sum(V &row_accum, const T &src) {
    row_reduce<base_ops::sum, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the product of each row of the src shared matrix in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void row_prod(V &row_accum, const T &src) {
    row_reduce<base_ops::mul, V, T, true>(row_accum, src, row_accum);
}

/**
 * @brief Store the maximum of each row of the src shared matrix, as well as the src_accum shared vector, in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void row_max(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::max, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the minimum of each row of the src shared matrix, as well as the src_accum shared vector, in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void row_min(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::min, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the sum of each row of the src shared matrix, as well as the src_accum shared vector, in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void row_sum(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::sum, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the product of each row of the src shared matrix, as well as the src_accum shared vector, in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void row_prod(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::mul, V, T, false>(row_accum, src, src_accum);
}

/**
 * @brief Store the maximum of each column of the src shared matrix in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void col_max(V &col_accum, const T &src) {
    col_reduce<base_ops::max, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the minimum of each column of the src shared matrix in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void col_min(V &col_accum, const T &src) {
    col_reduce<base_ops::min, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the sum of each column of the src shared matrix in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void col_sum(V &col_accum, const T &src) {
    col_reduce<base_ops::sum, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the product of each column of the src shared matrix in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void col_prod(V &col_accum, const T &src) {
    col_reduce<base_ops::mul, V, T, true>(col_accum, src, col_accum);
}

/**
 * @brief Store the maximum of each column of the src shared matrix, as well as the src_accum shared vector, in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void col_max(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::max, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the minimum of each column of the src shared matrix, as well as the src_accum shared vector, in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void col_min(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::min, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the sum of each column of the src shared tile, as well as the src_accum row vector, in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void col_sum(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::sum, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the product of each column of the src shared tile, as well as the src_accum row vector, in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void col_prod(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::mul, V, T, false>(col_accum, src, src_accum);
}

}
# 10 "/root/HipKittens//include/ops/warp/shared/tile/tile.cuh" 2
# 9 "/root/HipKittens//include/ops/warp/shared/shared.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/shared/vec/vec.cuh" 1
/**
 * @file
 * @brief An aggregate header for warp operations on data stored in shared memory.
 */



// #include "conversions.cuh"
# 1 "/root/HipKittens//include/ops/warp/shared/vec/maps.cuh" 1
/**
 * @file
 * @brief Warp-scope maps on shared vectors.
 */







namespace kittens {

/**
 * @brief Applies a unary operation to each element of a shared memory vector.
 *
 * @tparam op Unary operation type.
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector in which to store the result.
 * @param src[in] Source vector to apply the unary operation.
 */
template<typename op, ducks::sv::all T>
__attribute__((device)) static inline void unary_op(T &dst, const T &src) {
    __syncthreads();
#pragma unroll
    for(int cur = kittens::laneid(); cur < T::length; cur+=WARP_THREADS) {
        dst[cur] = op::template op<typename T::dtype>(src[cur]);
    }
}
/**
 * @brief Perform a binary operation on two shared vectors.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vectors.
 * @param dst[out] The destination vector where the result is stored.
 * @param lhs[in] The left-hand side vector for the operation.
 * @param rhs[in] The right-hand side vector for the operation.
 */
template<typename op, ducks::sv::all T>
__attribute__((device)) static inline void bin_op(T &dst, const T &lhs, const T &rhs) {
    __syncthreads();
#pragma unroll
    for(int cur = laneid(); cur < T::length; cur+=WARP_THREADS) {
        dst[cur] = op::template op<typename T::dtype>(lhs[cur], rhs[cur]);
    }
}
/**
 * @brief Perform a binary operation on a shared vector and a scalar.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vector.
 * @param dst[out] The destination vector where the result is stored.
 * @param src[in] The source vector for the operation.
 * @param param[in] The scalar parameter for the operation.
 */
template<typename op, ducks::sv::all T>
__attribute__((device)) static inline void bin_op(T &dst, const T &src, const typename T::dtype &param) {
    __syncthreads();
#pragma unroll
    for(int cur = laneid(); cur < T::length; cur+=WARP_THREADS) {
        dst[cur] = op::template op<typename T::dtype>(src[cur], param);
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// ---- const ops ----

/**
 * @brief Sets all elements of a shared memory vector to zero.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to zero.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void zero(T &dst) {
    unary_op<base_ops::zero, T>(dst, dst);
}
/**
 * @brief Sets all elements of a shared memory vector to one.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to one.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void one(T &dst) {
    unary_op<base_ops::one, T>(dst, dst);
}
/**
 * @brief Sets all elements of a shared memory vector to positive infinity.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to positive infinity.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void pos_infty(T &dst) {
    unary_op<base_ops::pos_infty, T>(dst, dst);
}
/**
 * @brief Sets all elements of a shared memory vector to negative infinity.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to negative infinity.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void neg_infty(T &dst) {
    unary_op<base_ops::neg_infty, T>(dst, dst);
}

// ---- unary ops ----

/**
 * @brief Copies the elements from one shared vector to another.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the source vector.
 * @param dst[out] Destination vector where the elements will be copied to.
 * @param src[in] Source vector to copy the elements from.
 */
template<ducks::sv::all T, typename U>
__attribute__((device)) static inline void copy(T &dst, const U &src) {
    bin_op<base_ops::copy2, T>(dst, dst, src); // the second arg is ignored here.
}
/**
 * @brief Applies the exponential function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void exp(T &dst, const T &src) {
    unary_op<base_ops::exp, T>(dst, src);
}
/**
 * @brief Applies the exponential function element-wise to a shared vector, in base 2.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void exp2(T &dst, const T &src) {
    unary_op<base_ops::exp2, T>(dst, src);
}
/**
 * @brief Applies the natural logarithm function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the logarithm values will be stored.
 * @param src[in] Source vector to apply the logarithm function to.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void log(T &dst, const T &src) {
    unary_op<base_ops::log, T>(dst, src);
}
/**
 * @brief Applies the logarithm base 2 function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the logarithm base 2 values will be stored.
 * @param src[in] Source vector to apply the logarithm base 2 function to.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void log2(T &dst, const T &src) {
    unary_op<base_ops::log2, T>(dst, src);
}
/**
 * @brief Applies the absolute value function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the absolute values will be stored.
 * @param src[in] Source vector to apply the absolute value function to.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void abs(T &dst, const T &src) {
    unary_op<base_ops::abs, T>(dst, src);
}
/**
 * @brief Applies the rectified linear unit (ReLU) function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the ReLU values will be stored.
 * @param src[in] Source vector to apply the ReLU function to.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void relu(T &dst, const T &src) {
    unary_op<base_ops::relu, T>(dst, src);
}

// ---- binary ops ----

/**
 * @brief Computes the element-wise maximum of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the maximum values will be stored.
 * @param lhs[in] First vector for the maximum operation.
 * @param rhs[in] Second vector for the maximum operation.
 */
template<ducks::sv::all T, typename U>
__attribute__((device)) static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::max, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise minimum of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the minimum values will be stored.
 * @param lhs[in] First vector for the minimum operation.
 * @param rhs[in] Second vector for the minimum operation.
 */
template<ducks::sv::all T, typename U>
__attribute__((device)) static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::min, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise sum of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the sum values will be stored.
 * @param lhs[in] First vector for the sum operation.
 * @param rhs[in] Second vector for the sum operation.
 */
template<ducks::sv::all T, typename U>
__attribute__((device)) static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sum, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise difference of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the difference values will be stored.
 * @param lhs[in] First vector for the difference operation.
 * @param rhs[in] Second vector for the difference operation.
 */
template<ducks::sv::all T, typename U>
__attribute__((device)) static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sub, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise product of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the product values will be stored.
 * @param lhs[in] First vector for the product operation.
 * @param rhs[in] Second vector for the product operation.
 */
template<ducks::sv::all T, typename U>
__attribute__((device)) static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::mul, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise division of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the division values will be stored.
 * @param lhs[in] First vector for the division operation.
 * @param rhs[in] Second vector for the division operation.
 */
template<ducks::sv::all T, typename U>
__attribute__((device)) static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::div, T>(dst, lhs, rhs);
}

}
# 10 "/root/HipKittens//include/ops/warp/shared/vec/vec.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/shared/vec/reductions.cuh" 1
/**
 * @file
 * @brief Warp-scope reductions on shared vectors.
 */






namespace kittens {

/**
 * @brief Performs a reduction operation on elements of a shared memory vector within a warp.
 *
 * This function applies a specified operation to reduce the elements of a shared memory vector `src` to a single value.
 * The result is stored in `accum`. If the `reset` parameter is true, the reduction includes an initial value `src_accum`.
 * The reduction operation is performed in a warp-wide context, ensuring synchronization between threads in the warp.
 *
 * @tparam op The operation to perform on the elements. Must provide a static `op` method.
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @tparam reset A boolean flag indicating whether to include an initial value in the reduction.
 * @param[out] accum The result of the reduction operation.
 * @param[in] src The shared memory vector to reduce.
 * @param[in] src_accum The initial value to include in the reduction if `reset` is false.
 */
template<typename op, ducks::sv::all SV, bool reset>
__attribute__((device)) static inline void reduce(typename SV::dtype &dst_accum, const SV &src, const typename SV::dtype &src_accum) {
    using T = SV::dtype;
    int laneid = kittens::laneid();
    T accum;
    if(laneid < src.length) accum = src[laneid]; // initialize a register accumulator
    __syncthreads();
    for(int i = laneid+kittens::WARP_THREADS; i < src.length; i+=kittens::WARP_THREADS) {
        accum = op::template op<T>(accum, src[i]);
    }
    __syncthreads();
    // We can now reduce within the warp.
    if (src.length > 32) {
        accum = op::template op<T>(accum, packed_shfl_down(kittens::MASK_ALL, accum, 32));
        __syncthreads();
    }
    if (src.length > 16) {
        accum = op::template op<T>(accum, packed_shfl_down(kittens::MASK_ALL, accum, 16));
        __syncthreads();
    }
    accum = op::template op<T>(accum, packed_shfl_down(kittens::MASK_ALL, accum, 8));
    __syncthreads();
    accum = op::template op<T>(accum, packed_shfl_down(kittens::MASK_ALL, accum, 4));
    __syncthreads();
    accum = op::template op<T>(accum, packed_shfl_down(kittens::MASK_ALL, accum, 2));
    __syncthreads();
    accum = op::template op<T>(accum, packed_shfl_down(kittens::MASK_ALL, accum, 1));
    __syncthreads();
    if constexpr (!reset) accum = op::template op<T>(accum, src_accum);
    // broadcast to all threads in the warp.
    dst_accum = packed_shfl(kittens::MASK_ALL, accum, 0); // everyone takes from warp leader
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

/**
 * @brief Finds the maximum element in a shared memory vector.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] max_val The maximum value found in the vector.
 * @param[in] src The shared memory vector to find the maximum in.
 */
template<ducks::sv::all SV>
__attribute__((device)) static inline void max(typename SV::dtype &max_val, const SV &src) {
    reduce<base_ops::max, SV, true>(max_val, src, max_val);
}

/**
 * @brief Finds the minimum element in a shared memory vector.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] min_val The minimum value found in the vector.
 * @param[in] src The shared memory vector to find the minimum in.
 */
template<ducks::sv::all SV>
__attribute__((device)) static inline void min(typename SV::dtype &min_val, const SV &src) {
    reduce<base_ops::min, SV, true>(min_val, src, min_val);
}

/**
 * @brief Calculates the sum of elements in a shared memory vector.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] sum_val The sum of the values in the vector.
 * @param[in] src The shared memory vector to sum.
 */
template<ducks::sv::all SV>
__attribute__((device)) static inline void sum(typename SV::dtype &sum_val, const SV &src) {
    reduce<base_ops::sum, SV, true>(sum_val, src, sum_val);
}

/**
 * @brief Calculates the product of elements in a shared memory vector.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] prod_val The product of the values in the vector.
 * @param[in] src The shared memory vector to multiply.
 */
template<ducks::sv::all SV>
__attribute__((device)) static inline void prod(typename SV::dtype &prod_val, const SV &src) {
    reduce<base_ops::mul, SV, true>(prod_val, src, prod_val);
}

// Three operand versions.

/**
 * @brief Finds the maximum element in a shared memory vector and accumulates it with src_accum.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] max_val The maximum value found in the vector, accumulated with src_accum.
 * @param[in] src The shared memory vector to find the maximum in.
 * @param[in] src_accum The initial value to accumulate with the maximum value found.
 */
template<ducks::sv::all SV>
__attribute__((device)) static inline void max(typename SV::dtype &max_val, const SV &src, const typename SV::dtype &src_accum) {
    reduce<base_ops::max, SV, false>(max_val, src, src_accum);
}

/**
 * @brief Finds the minimum element in a shared memory vector and accumulates it with src_accum.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] min_val The minimum value found in the vector, accumulated with src_accum.
 * @param[in] src The shared memory vector to find the minimum in.
 * @param[in] src_accum The initial value to accumulate with the minimum value found.
 */
template<ducks::sv::all SV>
__attribute__((device)) static inline void min(typename SV::dtype &min_val, const SV &src, const typename SV::dtype &src_accum) {
    reduce<base_ops::min, SV, false>(min_val, src, src_accum);
}

/**
 * @brief Calculates the sum of elements in a shared memory vector and accumulates it with src_accum.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] sum_val The sum of the values in the vector, accumulated with src_accum.
 * @param[in] src The shared memory vector to sum.
 * @param[in] src_accum The initial value to accumulate with the sum of the vector.
 */
template<ducks::sv::all SV>
__attribute__((device)) static inline void sum(typename SV::dtype &sum_val, const SV &src, const typename SV::dtype &src_accum) {
    reduce<base_ops::sum, SV, false>(sum_val, src, src_accum);
}

/**
 * @brief Calculates the product of elements in a shared memory vector and accumulates it with src_accum.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] prod_val The product of the values in the vector, accumulated with src_accum.
 * @param[in] src The shared memory vector to multiply.
 * @param[in] src_accum The initial value to accumulate with the product of the vector.
 */
template<ducks::sv::all SV>
__attribute__((device)) static inline void prod(typename SV::dtype &prod_val, const SV &src, const typename SV::dtype &src_accum) {
    reduce<base_ops::mul, SV, false>(prod_val, src, src_accum);
}
}
# 10 "/root/HipKittens//include/ops/warp/shared/vec/vec.cuh" 2
# 9 "/root/HipKittens//include/ops/warp/shared/shared.cuh" 2
# 13 "/root/HipKittens//include/ops/warp/warp.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/memory/memory.cuh" 1
/**
 * @file
 * @brief An aggregate header of warp memory operations, where a single warp loads or stores data on its own.
 */



# 1 "/root/HipKittens//include/ops/warp/memory/util/util.cuh" 1
/**
 * @file
 * @brief General memory utilities not specialized for either tiles or vectors.
 */






namespace kittens {

enum class coherency {
    cache_all = 0,
    cache_global = 1,
    cache_stream = 2,
    non_temporal = 3
};

/* ----------   Shared memory utilities  ---------- */
__attribute__((device)) inline float2 load_shared_vec(uint32_t lds_off) {
    float2 result;
    asm volatile(
        "ds_read_b64 %0, %1\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(result) // Output: store result in float2
        : "v"(lds_off) // Input: LDS offset to read from
        : "memory"
    );
    return result;
}

__attribute__((device)) inline void store_shared_vec(uint32_t lds_off, float2 val) {
    asm volatile(
        "ds_write_b64 %0, %1\n"
        :
        : "v"(lds_off), "v"(val)
        : "memory"
    );
}

__attribute__((device)) inline float2 load_global_vec2(const float2* gptr) {
    float2 v;
    // Use global_load_dwordx2 which is more cache-friendly than flat_load
    asm volatile(
        "global_load_dwordx2 %0, %1, off\n"
        "s_waitcnt vmcnt(0)\n"
        : "=v"(v)
        : "v"(gptr)
        : "memory"
    );
    return v;
}

__attribute__((device)) inline float4 load_global_vec4(const float4* gptr) {
    float4 v;
    // Use global_load_dwordx4 which is more cache-friendly than flat_load
    asm volatile(
        "global_load_dwordx4 %0, %1, off\n"
        "s_waitcnt vmcnt(0)\n"
        : "=v"(v)
        : "v"(gptr)
        : "memory"
    );
    return v;
}

using i32x4 = int32_t __attribute__((ext_vector_type(4)));
struct buffer_resource {
    uint64_t ptr;
    uint32_t range;
    uint32_t config;
};

__attribute__((device)) inline buffer_resource make_buffer_resource(uint64_t ptr, uint32_t range, uint32_t config) {
    return {ptr, range, config};
}

__attribute__((device)) inline i32x4 make_srsrc(const void* ptr, uint32_t range_bytes, uint32_t row_stride_bytes = 0) {
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(ptr); // width = sizeof(void*)
    std::uint64_t as_u64 = static_cast<std::uint64_t>(as_int); // widen if host is 32-bit
    buffer_resource rsrc = make_buffer_resource(as_u64, range_bytes, 0x110000);

    row_stride_bytes &= 0x3FFF;
    if (row_stride_bytes) {
        // - The swizzle stride lives in bits 13:0 of word2.
        //   Max value = 0x3FFF (8 KiB – one cache line per bank).
        uint64_t stride_field = row_stride_bytes;
        stride_field = stride_field | 0x4000; // Cache swizzle
        stride_field = stride_field | 0x8000; // Swizzle enable
        rsrc.ptr |= stride_field << 48;
    }

    return *reinterpret_cast<const i32x4*>(&rsrc);
}

__attribute__((device)) uint64_t llvm_amdgcn_raw_buffer_load_b64(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i64");

__attribute__((device)) __uint128_t llvm_amdgcn_raw_buffer_load_b128(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i128");

__attribute__((device)) void llvm_amdgcn_raw_buffer_store_b8(uint8_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i8");

__attribute__((device)) void llvm_amdgcn_raw_buffer_store_b16(uint16_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i16");

__attribute__((device)) void llvm_amdgcn_raw_buffer_store_b32(uint32_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i32");

__attribute__((device)) void llvm_amdgcn_raw_buffer_store_b64(uint64_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i64");

__attribute__((device)) void llvm_amdgcn_raw_buffer_store_b128(__uint128_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i128");


__attribute__((device)) inline float2 load_global_vec2_async(const float2* gptr) {
    float2 v;
    // Use global_load_dwordx2 which is more cache-friendly than flat_load
    asm volatile(
        "global_load_dwordx2 %0, %1, off\n"
        : "=v"(v)
        : "v"(gptr)
        : "memory"
    );
    return v;
}

__attribute__((device)) inline float4 load_global_vec4_async(const float4* gptr) {
    float4 v;
    // Use global_load_dwordx4 which is more cache-friendly than flat_load
    asm volatile(
        "global_load_dwordx4 %0, %1, off\n"
        : "=v"(v)
        : "v"(gptr)
        : "memory"
    );
    return v;
}

__attribute__((device)) inline void store_global_b128_async(void* gptr, __uint128_t value) {
    asm volatile(
        "global_store_dwordx4 %0, %1, off\n"
        :
        : "v"(gptr), "v"(value)
        : "memory"
    );
}

__attribute__((device)) inline float2 load_shared_vec_async(uint32_t lds_off) {
    float2 result;
    asm volatile(
        "ds_read_b64 %0, %1\n"
        // "s_waitcnt lgkmcnt(0)\n"
        : "=v"(result) // Output: store result in float2
        : "v"(lds_off) // Input: LDS offset to read from
        : "memory"
    );
    return result;
}

/* ----------   To prevent generic addressing  ---------- */

template<typename T> struct move {
    __attribute__((device)) static inline void lds(T& dst, uint32_t src);
    __attribute__((device)) static inline void sts(uint32_t dst, const T& src);
    __attribute__((device)) static inline void ldg(T& dst, T* src);
    __attribute__((device)) static inline void stg(T* dst, const T& src);
};

// meant to be used only with shared tiles and shared vectors
namespace detail {
template<typename T> struct size_info {
    static constexpr uint32_t bytes = sizeof(std::remove_reference_t<T>);
};
template<ducks::st::all ST> struct size_info<ST> {
    static constexpr uint32_t elements = ST::num_elements;
    static constexpr uint32_t bytes = ST::num_elements * sizeof(typename ST::dtype);
};
template<ducks::sv::all SV> struct size_info<SV> {
    static constexpr uint32_t elements = SV::length;
    static constexpr uint32_t bytes = SV::length * sizeof(typename SV::dtype);
};
}
template<typename... Args> inline constexpr uint32_t size_bytes = 0; // base case
template<typename T, typename... Args> inline constexpr uint32_t size_bytes<T, Args...> = detail::size_info<T>::bytes + size_bytes<Args...>; // recursive case

} // namespace kittens
# 9 "/root/HipKittens//include/ops/warp/memory/memory.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/memory/tile/tile.cuh" 1
/**
 * @file
 * @brief An aggregate header of warp memory operations on tiles, where a single warp loads or stores data on its own.
 */



# 1 "/root/HipKittens//include/ops/warp/memory/tile/shared_to_register.cuh" 1
/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */
# 14 "/root/HipKittens//include/ops/warp/memory/tile/shared_to_register.cuh"
namespace kittens {
// These probably need to be redone to reduce bank conflicts.
// They currently work fine with xor layout but it should be
// possible to reduce their bank conflicts with other layouts too.

/**
 * @brief Load data from a shared tile into a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 */
template<ducks::rt::all RT, ducks::st::all ST>
__attribute__((device)) inline static void load(RT &dst, const ST &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width == ST::width, "register tile and shared tile must match width");
    const int a = dst.width;
    using T2 = RT::dtype;
    using T = base_types::packing<T2>::unpacked_type;
    using U = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;

    const int laneid = kittens::laneid() % kittens::WARP_THREADS;
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);

    int row_offset, col_offset;
    if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        row_offset = laneid%16;
        col_offset = 4*(laneid/16);
    }
    else {
        row_offset = 4*(laneid/16);
        col_offset = laneid%16;
    }


#pragma unroll
    for(int j = 0; j < dst.width; j++) {
        const int col = j*dst.tile_size_col + col_offset;
        uint32_t addr = src.idx(src_ptr, {row_offset, col});
#pragma unroll
        for(int i = 0; i < dst.height; i++) {
            const int row = i*dst.tile_size_row + row_offset;
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) { // handle the row-major layout

                if constexpr (sizeof(typename ST::dtype) == 4) {
                    // handle float32
                    float2 loaded0 = load_shared_vec_sync(src.idx(src_ptr, {row, col}));
                    float2 loaded1 = load_shared_vec_sync(src.idx(src_ptr, {row, col+2}));
                    dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(loaded0);
                    dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(loaded1);
                } else {
                    // handle fp16 and bf16
                    if constexpr (sizeof(T) == sizeof(U)) {
                        // Same size: no conversion needed (e.g., bf16->bf16 or half->half)
                        // float2 loaded = load_shared_vec(src.idx(src_ptr, {row, col}));
                        // U2* tmp = reinterpret_cast<U2*>(&loaded);
                        // dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                        // dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);

                        // avoid v_bfi_b32
                        asm volatile(
                            "ds_read_b64 %0, %1 offset:%2\n"
                            : "=v"(*reinterpret_cast<uint64_t*>(&dst.tiles[i][j].data[0]))
                            : "v"(addr), "i"(i * ST::underlying_cols * kittens::TILE_ROW_DIM<U> * sizeof(U))
                            : "memory"
                        );
                    } else {
                        // Different size: converting bf16/half -> float
                        // Need to load and convert each pair separately
                        U2 loaded0 = *reinterpret_cast<const U2*>(&src[{row, col}]);
                        U2 loaded1 = *reinterpret_cast<const U2*>(&src[{row, col+2}]);
                        dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(loaded0);
                        dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(loaded1);
                    }
                }
            }
            else { // handle the column-major layout
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(U2{src[{row, col}], src[{row+1, col}]});
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(U2{src[{row+2, col}], src[{row+3, col}]});
            }
        }
    }
}


namespace zz2{
// template<ducks::rt::all RT, ducks::st::all ST>
using RT = rt<__hip_bfloat16, 64, 16, kittens::ducks::rt_layout::row>;
using ST = st_subtile<st<__hip_bfloat16, 256, 64>, 64, 16>;
__attribute__((device)) inline static void load(RT &dst, const ST &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width == ST::width, "register tile and shared tile must match width");

    using T2____hip_bfloat162 = RT::dtype;
    using T____hip_bfloat16 = base_types::packing<T2____hip_bfloat162>::unpacked_type;
    using __hip_bfloat16 = ST::dtype;
    using __hip_bfloat162 = base_types::packing<__hip_bfloat16 >::packed_type;

    const int laneid = kittens::laneid() % kittens::WARP_THREADS;
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);

    int row_offset, col_offset;
    if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        // True this ....
        row_offset = laneid%16;
        col_offset = 4*(laneid/16);
    }
    else {
        row_offset = 4*(laneid/16);
        col_offset = laneid%16;
    }


#pragma unroll
    for(int j = 0; j < dst.width; j++) { // 1
        const int col = j*dst.tile_size_col + col_offset;
        uint32_t addr = src.idx(src_ptr, {row_offset, col});
#pragma unroll
        for(int i = 0; i < dst.height; i++) { // 4
            const int row = i*dst.tile_size_row + row_offset;
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) { // handle the row-major layout

                if constexpr (sizeof(typename ST::dtype) == 4) { // false // 2 here....
                    // handle float32 // TODO: fix this function.
                    // float2 loaded0 = load_shared_vec_sync(src.idx(src_ptr, {row, col}));
                    // float2 loaded1 = load_shared_vec_sync(src.idx(src_ptr, {row, col+2}));
                    // dst.tiles[i][j].data[0] = base_types::convertor<T2____hip_bfloat162, __hip_bfloat162>::convert(loaded0);
                    // dst.tiles[i][j].data[1] = base_types::convertor<T2____hip_bfloat162, __hip_bfloat162>::convert(loaded1);
                } else {
                    // handle fp16 and bf16
                    if constexpr (sizeof(T____hip_bfloat16) == sizeof(__hip_bfloat16)) {
                        // Same size: no conversion needed (e.g., bf16->bf16 or half->half)
                        // float2 loaded = load_shared_vec(src.idx(src_ptr, {row, col}));
                        // U2* tmp = reinterpret_cast<U2*>(&loaded);
                        // dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                        // dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);

                        // avoid v_bfi_b32
                        asm volatile(
                            "ds_read_b64 %0, %1 offset:%2\n"
                            : "=v"(*reinterpret_cast<uint64_t*>(&dst.tiles[i][j].data[0]))
                            : "v"(addr), "i"(i * ST::underlying_cols * kittens::TILE_ROW_DIM<__hip_bfloat16> * sizeof(__hip_bfloat16))
                            : "memory"
                        );
                    } else {
                        // Different size: converting bf16/half -> float
                        // Need to load and convert each pair separately
                        __hip_bfloat162 loaded0 = *reinterpret_cast<const __hip_bfloat162*>(&src[{row, col}]);
                        __hip_bfloat162 loaded1 = *reinterpret_cast<const __hip_bfloat162*>(&src[{row, col+2}]);
                        dst.tiles[i][j].data[0] = base_types::convertor<T2____hip_bfloat162, __hip_bfloat162>::convert(loaded0);
                        dst.tiles[i][j].data[1] = base_types::convertor<T2____hip_bfloat162, __hip_bfloat162>::convert(loaded1);
                    }
                }
            }
            else { // handle the column-major layout
                dst.tiles[i][j].data[0] = base_types::convertor<T2____hip_bfloat162, __hip_bfloat162>::convert(__hip_bfloat162{src[{row, col}], src[{row+1, col}]});
                dst.tiles[i][j].data[1] = base_types::convertor<T2____hip_bfloat162, __hip_bfloat162>::convert(__hip_bfloat162{src[{row+2, col}], src[{row+3, col}]});
            }
        }
    }
}
}


/**
 * @brief Store data into a shared tile from a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 */
template<ducks::rt::all RT, ducks::st::all ST>
__attribute__((device)) inline static void store(ST &dst, const RT &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width == ST::width, "register tile and shared tile must match width");

    using T2 = RT::dtype;
    using T = base_types::packing<T2>::unpacked_type;
    using U = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;

    int laneid = kittens::laneid() % kittens::WARP_THREADS;
    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);

    int row_offset, col_offset;
    if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        row_offset = laneid%16;
        col_offset = 4*(laneid/16);
    }
    else {
        row_offset = 4*(laneid/16);
        col_offset = laneid%16;
    }
#pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = i*src.tile_size_row + row_offset;
#pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + col_offset;

            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) { // handle the row-major layout
                // *(U2*)(&dst[{row, col+0}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                // *(U2*)(&dst[{row, col+2}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                if constexpr (sizeof(typename ST::dtype) == 4) {
                    // handle float32
                    store_shared_vec(dst.idx(dst_ptr, {row, col}), base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]));
                    store_shared_vec(dst.idx(dst_ptr, {row, col+2}), base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]));
                } else {
                    // handle fp16 and bf16
                    if constexpr (sizeof(T) == sizeof(U)) {
                        // Same size: no conversion needed (e.g., bf16->bf16 or half->half)
                        float2 loaded = *reinterpret_cast<const float2*>(src.tiles[i][j].data);
                        store_shared_vec(dst.idx(dst_ptr, {row, col}), loaded);
                    } else {
                        // Different size: converting float -> bf16/half
                        // Need to convert and store each pair separately
                        U2 converted0 = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                        U2 converted1 = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                        *reinterpret_cast<U2*>(&dst[{row, col}]) = converted0;
                        *reinterpret_cast<U2*>(&dst[{row, col+2}]) = converted1;
                    }
                }
            }
            else { // handle the column-major layout
                U2 tmp[2];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);

                dst[{row+0, col}] = std::bit_cast<U>(tmp[0].x);
                dst[{row+1, col}] = std::bit_cast<U>(tmp[0].y);
                dst[{row+2, col}] = std::bit_cast<U>(tmp[1].x);
                dst[{row+3, col}] = std::bit_cast<U>(tmp[1].y);
            }
        }
    }
}



}
# 9 "/root/HipKittens//include/ops/warp/memory/tile/tile.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/memory/tile/global_to_register.cuh" 1
/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */







namespace kittens {

/**
 * @brief Load data from a source array into a row-major layout tile.
 *
 * @tparam RT The row-major layout tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param idx[in] The index of the tile to load data from.
 */
template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__attribute__((device)) inline static void load(RT &dst, const GL &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    int laneid = kittens::laneid();

    int row_offset = laneid%16, col_offset = 4*(laneid/16);

    uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(U);
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(src_ptr);
    std::uint64_t as_u64 = static_cast<std::uint64_t>(as_int); // widen if host is 32-bit
    buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);


#pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row = dst.tile_size_row*i + row_offset;

#pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = dst.tile_size_col*j + col_offset;

            U2* tmp;
            if constexpr (sizeof(U2) == 4) { // bf16_2
                float2 loaded = std::bit_cast<float2>(llvm_amdgcn_raw_buffer_load_b64(
                    std::bit_cast<i32x4>(br),
                    (row*row_stride + col) * sizeof(U),
                    0,
                    0
                ));
                tmp = reinterpret_cast<U2*>(&loaded);
            }
            else { // float2
                float4 loaded = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(
                    std::bit_cast<i32x4>(br),
                    (row*row_stride + col) * sizeof(U),
                    0,
                    0
                ));
                tmp = reinterpret_cast<U2*>(&loaded);
            }
#pragma unroll
            for(int k = 0; k < 2; k++) {
                dst.tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(tmp[k]);
            }
        }
    }
}


/**
 * @brief Load data from a source array into a column-major layout tile.
 *
 * @tparam RT The column-major layout tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param row_stride[in] The stride in elements between rows in the source array.
 */
template<int axis, ducks::rt::col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__attribute__((device)) inline static void load(RT &dst, const GL &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;

    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    int laneid = kittens::laneid();

    const int row_offset = 4*(laneid/16), col_offset = laneid%16;

#pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row = i*dst.tile_size_row + row_offset;

#pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + col_offset;

            dst.tiles[i][j].data[0].x = base_types::convertor<T, U>::convert(src_ptr[(row+0)*row_stride + col]);
            dst.tiles[i][j].data[0].y = base_types::convertor<T, U>::convert(src_ptr[(row+1)*row_stride + col]);
            dst.tiles[i][j].data[1].x = base_types::convertor<T, U>::convert(src_ptr[(row+2)*row_stride + col]);
            dst.tiles[i][j].data[1].y = base_types::convertor<T, U>::convert(src_ptr[(row+3)*row_stride + col]);
        }
    }

}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__attribute__((device)) inline static void load(RT &dst, const GL &src, const COORD &idx) {
    load<2, RT, GL>(dst, src, idx);
}

/**
 * @brief Store data from a register tile to a destination array in global memory with a row-major layout.
 *
 * @tparam RT The register tile type with a row-major layout.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__attribute__((device)) inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    int laneid = kittens::laneid();

    int row_offset = laneid%16, col_offset = 4*(laneid/16);

#pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = src.tile_size_row*i + row_offset;

#pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = src.tile_size_col*j + col_offset;

            U2 tmp[2];
#pragma unroll
            for(int k = 0; k < 2; k++) {
                tmp[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
            }
            if constexpr (sizeof(U2) == 4) { // bf16_2
                *(bytes_8*)&dst_ptr[row*row_stride + col] = *(bytes_8*)tmp;
            }
            else { // float2
                *(bytes_16*)&dst_ptr[row*row_stride + col] = *(bytes_16*)tmp;
            }
        }
    }
}


/**
 * @brief Store data from a register tile to a destination array in global memory with a column-major layout.
 *
 * @tparam RT The register tile type with a column-major layout.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<int axis, ducks::rt::col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__attribute__((device)) inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    const int laneid = kittens::laneid();

    const int row_offset = 4*(laneid/16), col_offset = laneid%16;

#pragma unroll
    for(int i = 0; i < src.height; i++) {
        const int row = i*src.tile_size_row + row_offset;

#pragma unroll
        for(int j = 0; j < src.width; j++) {
            const int col = j*src.tile_size_col + col_offset;
            dst_ptr[(row+0)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
            dst_ptr[(row+1)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
            dst_ptr[(row+2)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
            dst_ptr[(row+3)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);
        }
    }
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__attribute__((device)) inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    store<2, RT, GL, COORD>(dst, src, idx);
}

}
# 10 "/root/HipKittens//include/ops/warp/memory/tile/tile.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/memory/tile/global_to_shared.cuh" 1
/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */






namespace kittens {

template< int axis, bool assume_aligned,
          ducks::st::all ST, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<ST>,
          int N_THREADS = WARP_THREADS >
__attribute__((device)) inline void load(ST& dst, const GL& src, const COORD& idx)
{
    using T = typename ST::dtype;
    const int row_stride = src.template stride<axis>();
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype); // if bf16, then 16/2 = 8. if fp8, then 16/1 = 16.
    constexpr int elem_per_half_memcpy = sizeof(float2)/sizeof(typename ST::dtype); // if bf16, then 8/2 = 4. if fp8, then 8/1 = 8.
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy; // if 64 columns, then 64/8 = 8 or 64/16 = 4
    constexpr int total_calls = (ST::cols * ST::rows + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];

    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int laneid = threadIdx.x % N_THREADS;

    // TODO: This is a hack to avoid the issue of too many VGPRs.
    // We should find a better way to do this.
    const int small_calls = 16;
    const int big_calls = (total_calls + small_calls - 1) / small_calls;
    float4 buf[small_calls];

    for (int i = 0; i < big_calls; i++) {
        const int offset = i * small_calls;
#pragma unroll
        for(int j = 0; j < small_calls; j++) {
            int load_idx = (offset + j) * N_THREADS + laneid;
            int row = load_idx / memcpy_per_row;
            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;

            if (row < dst.rows) {
                buf[j] = load_global_vec4_async((float4*) (src_ptr + (row * row_stride + col))); // thread loads 128-bits, 16-bytes
            }
        }




        asm volatile("s_waitcnt vmcnt(0)");


#pragma unroll
        for(int j = 0; j < small_calls; j++) {
            int load_idx = (offset + j) * N_THREADS + laneid;
            int row = load_idx / memcpy_per_row;
            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;

            if (row < dst.rows) {
                store_shared_vec(dst.idx(dst_ptr, {row, col}), {buf[j].x, buf[j].y});
                store_shared_vec(dst.idx(dst_ptr, {row, col + elem_per_half_memcpy}), {buf[j].z, buf[j].w});
            }
        }




        asm volatile("s_waitcnt lgkmcnt(0)");

    }
}

template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__attribute__((device)) static inline void load(ST &dst, const GL &src, const COORD &idx) {
    load<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}


/********************************************* Register Pipelining ************************************************** */

/**
 * @brief Load from global memory to registers with proper batching for cache locality
 *
 * @tparam reg_buffer The register buffer to store data into.
 * @tparam U The data type of the destination array.
 * @param[out] reg_buffer The register buffer to store data into.
 * @param[in] buffer_size The size of the register buffer.
 * @param[in] src The source global memory array to store data from.
 * @param[in] idx The index into the source global memory array.
 * @param[in] dst_template The template of the ultimate shared tile that will be loaded into.
 */
template<int axis=2, bool assume_aligned=false,
        int N_THREADS = WARP_THREADS,
        ducks::st::all ST,
        ducks::gl::all GL,
        ducks::coord::tile COORD = coord<ST>
>
__attribute__((device)) inline void load_global_to_register_buffer(float4* reg_buffer, const int buffer_size, const GL& src, const COORD& idx, const ST& dst_template) {
    using T = typename ST::dtype;
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(T);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_chunks = (ST::rows * ST::cols) / elem_per_memcpy;
    constexpr int total_calls = (total_chunks + N_THREADS - 1) / N_THREADS;
    constexpr int small_calls = 16;
    const int big_calls = (total_calls + small_calls - 1) / small_calls;

    const int row_stride = src.template stride<axis>();
    const int row_stride_bytes = row_stride * sizeof(T);
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* base_ptr = (T*)&src[unit_coord]; // global memory pointer
    const int laneid = threadIdx.x % N_THREADS;

    // buffer resource
    const int total_bytes = row_stride * ST::rows * sizeof(T);
    i32x4 srsrc = make_srsrc(base_ptr, total_bytes, row_stride_bytes);

    int buf_idx = 0;
    for (int i = 0; i < big_calls && buf_idx < buffer_size; ++i) {
        const int offset = i * small_calls;
#pragma unroll
        for (int j = 0; j < small_calls; ++j) {
            const int chunk_idx = (offset + j) * N_THREADS + laneid;
            if (chunk_idx < total_chunks && buf_idx < buffer_size) {
                int row = chunk_idx / memcpy_per_row;
                int col = (chunk_idx % memcpy_per_row) * elem_per_memcpy;
                int flat_offset = row * row_stride + col;
                int byte_offset = flat_offset * sizeof(T);
                __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, byte_offset, 0, 0);
                reg_buffer[buf_idx] = *reinterpret_cast<float4*>(&raw);
                buf_idx++;
            }
        }
    }
}
namespace zz{
    using GL = gl<__hip_bfloat16, -1, -1, -1, -1>;
    using ST = st<__hip_bfloat16, 256, 64>;
    using COORD = coord<ST>;
    constexpr int N_THREADS = 512;
    constexpr int axis = 2;
    __attribute__((device)) inline void load_global_to_register_buffer(float4* reg_buffer, const int buffer_size, const GL& src, const COORD& idx, const ST& dst_template) {
        constexpr int elem_per_memcpy_8 = sizeof(float4)/sizeof(__hip_bfloat16);
        constexpr int memcpy_per_row_8 = ST::cols / elem_per_memcpy_8;
        constexpr int total_chunks_2048 = (ST::rows * ST::cols) / elem_per_memcpy_8;
        constexpr int total_calls_4 = (total_chunks_2048 + N_THREADS - 1) / N_THREADS;
        constexpr int small_calls = 16;
        const int big_calls_1 = (total_calls_4 + small_calls - 1) / small_calls;

        const int row_stride /*cols so 8192 here..*/ = src.template stride<axis>();
        const int row_stride_bytes = row_stride * sizeof(__hip_bfloat16);
        coord<> unit_coord = idx.template unit_coord<axis, 3>();
        __hip_bfloat16* base_ptr = (__hip_bfloat16*)&src[unit_coord]; // global memory pointer
        const int laneid = threadIdx.x % N_THREADS;

        // buffer resource
        const int total_bytes = row_stride * ST::rows * sizeof(__hip_bfloat16);
        i32x4 srsrc = make_srsrc(base_ptr, total_bytes, row_stride_bytes);

        int buf_idx = 0;
        for (int i = 0; i < big_calls_1 && buf_idx < buffer_size; ++i) {
            const int offset = i * small_calls;
#pragma unroll
            for (int j = 0; j < small_calls; ++j) {
                const int chunk_idx = (offset + j) * N_THREADS + laneid;
                if (chunk_idx < total_chunks_2048 && buf_idx < buffer_size) {
                    int row = chunk_idx / memcpy_per_row_8;
                    int col = (chunk_idx % memcpy_per_row_8) * elem_per_memcpy_8;
                    int flat_offset = row * row_stride + col;
                    int byte_offset = flat_offset * sizeof(__hip_bfloat16);
                    __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, byte_offset, 0, 0);
                    reg_buffer[buf_idx] = *reinterpret_cast<float4*>(&raw);
                    buf_idx++;
                }
            }
        }
    }
    } // namespace zz
/**
 * @brief Store from registers to shared memory (preserving the batched pattern)
 *
 * @tparam reg_buffer The register buffer to store data into.
 * @tparam ST The type of the destination shared tile.
 * @param[out] dst The destination shared tile to store data into.
 * @param[in] reg_buffer The register buffer to store data from.
 */
template<int N_THREADS = WARP_THREADS, ducks::st::all ST>
__attribute__((device)) inline void store_register_buffer_to_shared(ST& dst, const float4* reg_buffer) {
    using T = typename ST::dtype;
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(T);
    constexpr int elem_per_half_memcpy = sizeof(float2)/sizeof(T);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;

    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int laneid = threadIdx.x % N_THREADS;

    constexpr int total_chunks = (ST::rows * ST::cols) / elem_per_memcpy;
    constexpr int total_calls = (total_chunks + N_THREADS - 1) / N_THREADS;
    constexpr int small_calls = 16;
    const int big_calls = (total_calls + small_calls - 1) / small_calls;

    int buf_idx = 0;
    // Store in the same batched pattern to maintain locality
#pragma unroll
    for (int i = 0; i < big_calls; i++) {
        const int offset = i * small_calls;
#pragma unroll
        for(int j = 0; j < small_calls; j++) {
            int load_idx = (offset + j) * N_THREADS + laneid;
            int row = load_idx / memcpy_per_row;
            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;
            if (row < dst.rows && buf_idx < 64) { // Safety check - use fixed limit
                const float4& buf_val = reg_buffer[buf_idx];
                store_shared_vec(dst.idx(dst_ptr, {row, col}), {buf_val.x, buf_val.y});
                store_shared_vec(dst.idx(dst_ptr, {row, col + elem_per_half_memcpy}), {buf_val.z, buf_val.w});
                buf_idx++;
            }
        } // Wait for this batch of stores to complete
    }
}



/******************************************************************************************************************** */



/**
 * @brief Stores data from a shared memory tile into global memory.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride[in] The stride between rows in the destination array.
 */
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>, int N_THREADS=WARP_THREADS>
__attribute__((device)) static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    using T = typename ST::dtype;
    const int row_stride = dst.template stride<axis>();
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int elem_per_float = sizeof(float)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::cols * ST::rows + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst[unit_coord];

    uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    int laneid = threadIdx.x % N_THREADS;

#pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int load_idx = i * N_THREADS + laneid;
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % src.cols;

        if (row < src.rows) {
            *(float*) &dst_ptr[row * row_stride + col] = *(float*)(&src[{row, col}]);
            *(float*) &dst_ptr[row * row_stride + col + elem_per_float] = *(float*)(&src[{row, col + elem_per_float}]);
            *(float*) &dst_ptr[row * row_stride + col + elem_per_float * 2] = *(float*)(&src[{row, col + elem_per_float * 2}]);
            *(float*) &dst_ptr[row * row_stride + col + elem_per_float * 3] = *(float*)(&src[{row, col + elem_per_float * 3}]);
        }
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__attribute__((device)) static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    store<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}
}
# 11 "/root/HipKittens//include/ops/warp/memory/tile/tile.cuh" 2
# 10 "/root/HipKittens//include/ops/warp/memory/memory.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/memory/vec/vec.cuh" 1
/**
 * @file
 * @brief An aggregate header of warp memory operations on vectors, where a single warp loads or stores data on its own.
 */



# 1 "/root/HipKittens//include/ops/warp/memory/vec/shared_to_register.cuh" 1
/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */
# 14 "/root/HipKittens//include/ops/warp/memory/vec/shared_to_register.cuh"
namespace kittens {

/**
 * @brief Load data from a shared vector into a register vector.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination register vector.
 * @param src[in]  The source shared vector.
 */
template<ducks::rv::all RV, ducks::sv::all SV>
__attribute__((device)) inline static void load(RV &dst, const SV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(SV::length == RV::length);

    int laneid = ::kittens::laneid();

    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
#pragma unroll
        for(auto w = 0; w < (dst.outer_dim+3)/4; w++) {
            int idx = w*128 + 2 * laneid;
            int o_dim = w*4 + (laneid/8) / 2;
            int i_dim = (laneid/8) % 2;
            // this should be a maximally coalesced load.
            if(idx < dst.length) {
                dst[o_dim][i_dim] = base_types::convertor<T2, U2>::convert(*(U2*)&src.data[idx]);
            }
        }
        // now we need to do a bunch of shuffle_sync's to make sure everyone has everything they need.
#pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int leader = 16*(w%4) + (laneid%8); // repeats every 128 columns
            dst[w][0] = packed_shfl(MASK_ALL, dst[w][0], leader);
            dst[w][1] = packed_shfl(MASK_ALL, dst[w][1], leader+8);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
        // otherwise there will be some pain :/
#pragma unroll
        for(auto w = 0; w < (dst.outer_dim+3)/4; w++) {
            int idx = w*64 + (laneid%8)*8 + (laneid/8);
            int o_dim = w*2 + (laneid%4) / 2;
            // this should be a maximally coalesced load.
            if(idx < dst.length) {
                T tmp = base_types::convertor<T, U>::convert(src.data[idx]);
                dst[o_dim][0] = tmp;
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
#pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int idx = w*64 + laneid;
            if(idx < dst.length) {
                dst[w][0] = base_types::convertor<T, U>::convert(src.data[idx]);
            }
        }
    }
}

/**
 * @brief Store data into a shared vector from a register vector.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination shared vector.
 * @param src[in]  The source register vector.
 */
template<ducks::sv::all SV, ducks::rv::all RV>
__attribute__((device)) inline static void store(SV &dst, const RV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(SV::length == RV::length);

    int laneid = ::kittens::laneid();

    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
#pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            int idx = w*128 + 2 * laneid;
            int o_dim = w*4 + (laneid/8) / 2;
            int i_dim = (laneid/8) % 2;
            // this should be a maximally coalesced store. I hope!
            if(idx < src.length)
                *(U2*)&dst.data[idx] = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
        // otherwise there will be some pain :/
#pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            int idx = w*64 + (laneid%8)*8 + (laneid/8);
            int o_dim = w*2 + (laneid%4) / 2;
            // this should be a maximally coalesced load.
            if(idx < src.length) {
                dst.data[idx] = base_types::convertor<U, T>::convert(src[o_dim][0]);
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
#pragma unroll
        for(auto w = 0; w < src.outer_dim; w++) {
            int idx = w*64 + laneid;
            if(idx < src.length) {
                dst.data[idx] = base_types::convertor<U, T>::convert(src[w][0]);
            }
        }
    }
}

}
# 9 "/root/HipKittens//include/ops/warp/memory/vec/vec.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/memory/vec/global_to_register.cuh" 1
/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */






namespace kittens {

 /**
 * @brief Load data into a register vector from a source array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the source array.
 * @param[out] dst The destination register vector to load data into.
 * @param[in] src The source array in global memory to load data from.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__attribute__((device)) inline static void load(RV &dst, const GL &src, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    U *src_ptr = (U*)&src[(idx.template unit_coord<-1, 3>())];
    int laneid = ::kittens::laneid();

    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
#pragma unroll
        for(auto w = 0; w < (dst.outer_dim+3)/4; w++) {
            int idx = w*64 + 2 * laneid;
            int o_dim = w*4 + (laneid/8) / 2;
            int i_dim = (laneid/8) % 2;
            // this should be a maximally coalesced load.
            if(idx < dst.length)
                dst[o_dim][i_dim] = base_types::convertor<T2, U2>::convert(*(U2*)&src_ptr[idx]);
        }
        // now we need to do a bunch of shuffle_sync's to make sure everyone has everything they need.
#pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int leader = 16*(w%4) + (laneid%8); // repeats every 128 columns
            dst[w][0] = packed_shfl(MASK_ALL, dst[w][0], leader);
            dst[w][1] = packed_shfl(MASK_ALL, dst[w][1], leader+8);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
#pragma unroll
        for(auto w = 0; w < (dst.outer_dim+3)/4; w++) {
            int idx = w*64 + (laneid%8)*8 + (laneid/8);
            int o_dim = w*2 + (laneid%4) / 2;
            // this should be a maximally coalesced load.
            if(idx < dst.length) {
                T tmp = base_types::convertor<T, U>::convert(src_ptr[idx]);
                dst[o_dim][0] = tmp;
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
#pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int idx = w*64 + laneid;
            if(idx < dst.length) {
                dst[w][0] = base_types::convertor<T, U>::convert(src_ptr[idx]);
            }
        }
    }
}

/**
 * @brief Store data from a register vector to a destination array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register vector to store data from.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__attribute__((device)) inline static void store(const GL &dst, const RV &src, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<-1, 3>())];
    int laneid = ::kittens::laneid();

    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
#pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            int idx = w*64 + 2 * laneid;
            int o_dim = w*4 + (laneid/8) / 2;
            int i_dim = (laneid/8) % 2;
            // this should be a maximally coalesced store. I hope! 
            if(idx < src.length) *(U2*)&dst_ptr[idx] = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
        // otherwise there will be some pain :/
#pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            int idx = w*64 + (laneid%8)*8 + (laneid/8);
            int o_dim = w*2 + (laneid%4) / 2;
            // this should be a maximally coalesced load.
            if(idx < src.length) {
                U tmp;
                tmp = base_types::convertor<U, T>::convert(src[o_dim][0]);
                dst_ptr[idx] = tmp;
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
#pragma unroll
        for(auto w = 0; w < src.outer_dim; w++) {
            int idx = w*64 + laneid;
            if(idx < src.length) {
                dst_ptr[idx] = base_types::convertor<U, T>::convert(src[w][0]);
            }
        }
    }
}
} // namespace kittens
# 10 "/root/HipKittens//include/ops/warp/memory/vec/vec.cuh" 2
# 1 "/root/HipKittens//include/ops/warp/memory/vec/global_to_shared.cuh" 1
/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */






namespace kittens {

/**
 * @brief Loads data from global memory into a shared memory vector.
 *
 * @tparam ST The shared memory vector type.
 * @param[out] dst The destination shared memory vector.
 * @param[in] src The source global memory array.
 * @param[in] idx The coord of the global memory array.
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__attribute__((device)) static inline void load(SV &dst, const GL &src, const COORD &idx) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = (SV::length + WARP_THREADS*elem_per_transfer - 1) / (WARP_THREADS*elem_per_transfer); // round up
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[(idx.template unit_coord<-1, 3>())];
#pragma unroll
    for(int iter = 0, i = ::kittens::laneid(); iter < total_calls; iter++, i+=WARP_THREADS) {
        if(i * elem_per_transfer < SV::length) {
            *(float4*)&dst.data[i*elem_per_transfer] = *(float4*)&src_ptr[i*elem_per_transfer];
        }
    }
}

/**
 * @brief Stores data from a shared memory vector into global memory.
 *
 * @tparam ST The shared memory vector type.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory vector.
 * @param[in] idx The coord of the global memory array.
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__attribute__((device)) static inline void store(const GL &dst, const SV &src, const COORD &idx) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = (SV::length + WARP_THREADS*elem_per_transfer-1) / (WARP_THREADS*elem_per_transfer); // round up
    typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst[(idx.template unit_coord<-1, 3>())];
#pragma unroll
    for(int iter = 0, i = ::kittens::laneid(); iter < total_calls; iter++, i+=WARP_THREADS) {
        if(i * elem_per_transfer < SV::length) {
            *(float4*)&dst_ptr[i*elem_per_transfer] = *(float4*)&src.data[i*elem_per_transfer];
        }
    }
}

}
# 10 "/root/HipKittens//include/ops/warp/memory/vec/vec.cuh" 2
# 10 "/root/HipKittens//include/ops/warp/memory/memory.cuh" 2
# 13 "/root/HipKittens//include/ops/warp/warp.cuh" 2
# 9 "/root/HipKittens//include/ops/ops.cuh" 2
# 1 "/root/HipKittens//include/ops/group/group.cuh" 1
/**
 * @file
 * @brief An aggregate header of all group (multi-warp) operations defined by ThunderKittens
 */







// A "warpgroup" is a special group of 4 consecutive warps defined by NVIDIA for certain SM_90+ operations.


namespace kittens {
/*
This is meant to be used with a `using group_N = kittens::group<NUM_WORKERS>;` at the start of every kernel.
*/
template<int N_WARPS>
struct group {
static constexpr int GROUP_WARPS = N_WARPS; // This alias produces nice parallelism.
static constexpr int GROUP_THREADS = N_WARPS * kittens::WARP_THREADS; // This alias produces nice parallelism.
__attribute__((device)) static inline int laneid() { return threadIdx.x % GROUP_THREADS; }
__attribute__((device)) static inline int warpid() { return laneid() / kittens::WARP_THREADS; }
__attribute__((device)) static inline int groupid() { return threadIdx.x / GROUP_THREADS; }

# 1 "/root/HipKittens//include/ops/group/memory/memory.cuh" 1
/**
 * @file
 * @brief An aggregate header of colaborative group memory movement operations
 */

# 1 "/root/HipKittens//include/ops/group/memory/tile/tile.cuh" 1
/**
 * @file
 * @brief An aggregate header of group memory operations on tiles.
 */

# 1 "/root/HipKittens//include/ops/group/memory/tile/shared_to_register.cuh" 1
/**
 * @file
 * @brief Functions for a warpgroup to collaboratively transfer data directly between shared memory and registers and back.
 */

/**
 * @brief Collaboratively load data from a shared tile into register tiles split across a warpgroup.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 */
template<ducks::rt::all RT, ducks::st::all ST>
__attribute__((device)) inline static void load(RT &dst, const ST &src) {
    constexpr int height = ST::height;
    constexpr int warp_height = RT::height;
    static_assert(height%N_WARPS == 0, "Group load / store requires tile height to be a multiple of N_WARPS.");
    static_assert(height%warp_height == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(ST::width==RT::width, "Group load / store requires tile widths to match.");
    int local_warpid = warpid();
    using T2 = RT::dtype;
    using U = ST::dtype;
    using T = base_types::packing<T2>::unpacked_type;
    using U2 = base_types::packing<U>::packed_type;
    int warp_laneid = ::kittens::laneid();

    int warp_row_offset = local_warpid * warp_height;
    int row_offset, col_offset;
    if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        row_offset = warp_laneid%16;
        col_offset = 4*(warp_laneid/16);
    }
    else {
        row_offset = 4*(warp_laneid/16);
        col_offset = warp_laneid%16;
    }

#pragma unroll
    for (int i = 0; i < dst.height; i++) {
        int row = (warp_row_offset + i) * dst.tile_size_row + row_offset;
#pragma unroll
        for (int j = 0; j < dst.width; j++) {
            int col = j * dst.tile_size_col + col_offset;
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) { // handle the row-major layout
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(*(U2*)(&src.data[row * src.underlying_rows + col]));
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(*(U2*)(&src.data[row * src.underlying_rows + col + 2]));
            }
            else { // handle the column-major layout
                U2 tmp[2];

                tmp[0] = U2{*(U*)(&src.data[row * src.underlying_rows + col]), *(U*)(&src.data[(row + 1) * src.underlying_rows + col]) };
                tmp[1] = U2{*(U*)(&src.data[(row + 2) * src.underlying_rows + col]), *(U*)(&src.data[(row + 3) * src.underlying_rows + col]) };

                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
            }
        }
    }
}


/**
 * @brief Collaboratively store data into a shared tile from register tiles split across a warpgroup.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 */
template<ducks::st::all ST, ducks::rt::all RT>
__attribute__((device)) inline static void store(ST &dst, const RT &src) {
    constexpr int height = ST::height;
    constexpr int warp_height = RT::height;
    static_assert(height%N_WARPS == 0, "Group load / store requires tile height to be a multiple of N_WARPS.");
    static_assert(height%warp_height == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(ST::width==RT::width, "Group load / store requires tile widths to match.");
    int local_warpid = warpid();
    using T2 = RT::dtype;
    using U = ST::dtype;
    using T = base_types::packing<T2>::unpacked_type;
    using U2 = base_types::packing<U>::packed_type;
    int warp_laneid = ::kittens::laneid();

    int warp_row_offset = local_warpid * warp_height;
    int row_offset, col_offset;
    if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        row_offset = warp_laneid%16;
        col_offset = 4*(warp_laneid/16);
    }
    else {
        row_offset = 4*(warp_laneid/16);
        col_offset = warp_laneid%16;
    }
#pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = (warp_row_offset + i) * src.tile_size_row + row_offset;
#pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j * src.tile_size_col + col_offset;
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) { // handle the row-major layout
                *(U2*)(&dst.data[row * dst.underlying_rows + col]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                *(U2*)(&dst.data[row * dst.underlying_rows + col + 2]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
            }
            else { // handle the column-major layout
                U2 tmp[2];

                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);

                *(U*)(&dst.data[row * dst.underlying_rows + col]) = std::bit_cast<U>(tmp[0].x);
                *(U*)(&dst.data[(row + 1) * dst.underlying_rows + col]) = std::bit_cast<U>(tmp[0].y);
                *(U*)(&dst.data[(row + 2) * dst.underlying_rows + col]) = std::bit_cast<U>(tmp[1].x);
                *(U*)(&dst.data[(row + 3) * dst.underlying_rows + col]) = std::bit_cast<U>(tmp[1].y);
            }
        }
    }
}
# 7 "/root/HipKittens//include/ops/group/memory/tile/tile.cuh" 2
# 1 "/root/HipKittens//include/ops/group/memory/tile/global_to_shared.cuh" 1
/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared tiles from and storing to global memory. 
 */

template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__attribute__((device)) static inline void load(ST &dst, const GL &src, const COORD &idx) {
    kittens::load<axis, assume_aligned, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>> // default case
__attribute__((device)) static inline void load(ST &dst, const GL &src, const COORD &idx) {
    kittens::load<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__attribute__((device)) static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    kittens::store<axis, assume_aligned, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>> // default case
__attribute__((device)) static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    kittens::store<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
# 8 "/root/HipKittens//include/ops/group/memory/tile/tile.cuh" 2
# 7 "/root/HipKittens//include/ops/group/memory/memory.cuh" 2
# 28 "/root/HipKittens//include/ops/group/group.cuh" 2
# 1 "/root/HipKittens//include/ops/group/shared/shared.cuh" 1
/**
 * @file
 * @brief An aggregate header of group operations on data in shared memory
 */

# 1 "/root/HipKittens//include/ops/group/shared/tile/tile.cuh" 1
/**
 * @file
 * @brief An aggregate header for group operations on shared tiles.
 */

# 1 "/root/HipKittens//include/ops/group/shared/tile/conversions.cuh" 1
/**
 * @file
 * @brief Group conversions between different shared memory tile types.
 */

/* ----------  COPIES  ---------- */

/**
 * @brief Copies data from one shared memory tile to another, potentially with different data types and layouts.
 *
 * @tparam T The data type of the destination tile.
 * @tparam U The data type of the source tile.
 * @tparam _height The height of the tile.
 * @tparam _width The width of the tile.
 * @tparam L1 The layout of the destination tile.
 * @tparam L2 The layout of the source tile.
 * @param[out] dst The destination tile.
 * @param[in] src The source tile.
 */
template<typename T, typename U, int _height, int _width>
__attribute__((device)) static inline void copy(st<T, _height, _width> &dst, const st<U, _height, _width> &src) {
#pragma unroll
    for(int i = laneid(); i < dst.num_elements; i+=GROUP_THREADS) {
        int row = i/dst.cols, col = i%dst.cols;
        dst[{row, col}] = base_types::convertor<T, U>::convert(src[{row, col}]);
    }
}
# 7 "/root/HipKittens//include/ops/group/shared/tile/tile.cuh" 2
# 1 "/root/HipKittens//include/ops/group/shared/tile/maps.cuh" 1
/**
 * @file
 * @brief Group maps on shared tiles.
 */

/**
 * @brief Performs a uniform unary operation on a tile.
 * 
 * This function applies a given unary operation to each element of the source tile and stores the result in the destination tile.
 * The operation is applied independently to each element, without considering its position or the values of neighboring elements.
 * 
 * @tparam op The unary operation to be applied. Must be specialized to support operation on the data type of T.
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the unary operation is applied.
 */
template<typename op, ducks::st::all T> // T2, w, h can be inferred from dst as long as op is specialized
__attribute__((device)) static inline void unary_map(T &dst, const T &src) {
#pragma unroll
    for(int i = laneid(); i < dst.num_elements; i += GROUP_THREADS) {
        dst.data[i] = op::template op<typename T::dtype>(src.data[i]);
    }
}

/**
 * @brief Performs a uniform binary operation on a tile with a scalar parameter.
 * 
 * This function applies a given binary operation to each element of the source tile and a scalar parameter, then stores the result in the destination tile.
 * The operation is applied independently to each element, treating the scalar parameter as the second operand for each operation.
 * 
 * @tparam op The binary operation to be applied. Must be specialized to support operation on the data type of T and the scalar parameter.
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the binary operation is applied.
 * @param[in] param The scalar parameter to be used as the second operand in the binary operation.
 */
template<typename op, ducks::st::all T>
__attribute__((device)) static inline void bin_map(T &dst, const T &src, const typename T::dtype &param) {
#pragma unroll
    for(int i = laneid(); i < dst.num_elements; i += GROUP_THREADS) {
        dst.data[i] = op::template op<typename T::dtype>(src.data[i], param);
    }
}

/**
 * @brief Performs a uniform binary operation on two tiles.
 * 
 * This function applies a given binary operation to corresponding elements of two source tiles and stores the result in the destination tile.
 * The operation is applied independently to each pair of elements, without considering their positions or the values of neighboring elements.
 * 
 * @tparam op The binary operation to be applied. Must be specialized to support operation on the data type of T.
 * @tparam T The type of the tiles. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile to which the binary operation is applied.
 * @param[in] rhs The second source tile to which the binary operation is applied.
 */
template<typename op, ducks::st::all T>
__attribute__((device)) static inline void bin_map(T &dst, const T &lhs, const T &rhs) {
#pragma unroll
    for(int i = laneid(); i < dst.num_elements; i += GROUP_THREADS) {
        dst.data[i] = op::template op<typename T::dtype>(lhs.data[i], rhs.data[i]);
    }
}

/**
 * @brief Performs a row-wise binary operation on a tile with a vector.
 * 
 * This function applies a given binary operation to each row of the source tile and the corresponding element of the source vector,
 * then stores the result in the destination tile. The operation is applied independently to each row, using the vector element as 
 * the second operand for each element in the row.
 * 
 * @tparam op The binary operation to be applied. Must be specialized to support operation on the data type of T and the vector elements.
 * @tparam T The type of the tiles. Must satisfy the `ducks::st::all` concept.
 * @tparam V The type of the vector. Must have the same data type as T.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the binary operation is applied.
 * @param[in] vec The source vector containing the second operand for each row operation.
 */
template<typename op, ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void row_map(T &dst, const T &src, const V &vec) {
    static_assert(std::is_same<typename T::dtype, typename V::dtype>::value, "Tile and vector must have the same data type");
    static_assert(V::length == T::rows, "Vector length must match the number of rows in the tile");
#pragma unroll
    for(int i = laneid(); i < dst.num_elements; i += GROUP_THREADS) {
        int row = i/dst.cols, col = i%dst.cols;
        dst[{row, col}] = op::template op<typename T::dtype>(src[{row, col}], vec[row]);
    }
}

/**
 * @brief Performs a column-wise binary operation on a tile with a vector.
 * 
 * This function applies a given binary operation to each column of the source tile and the corresponding element of the source vector,
 * then stores the result in the destination tile. The operation is applied independently to each column, using the vector element as 
 * the second operand for each element in the column.
 * 
 * @tparam op The binary operation to be applied. Must be specialized to support operation on the data type of T and the vector elements.
 * @tparam T The type of the tiles. Must satisfy the `ducks::st::all` concept.
 * @tparam V The type of the vector. Must have the same data type as T.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the binary operation is applied.
 * @param[in] vec The source vector containing the second operand for each column operation.
 */
template<typename op, ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void col_map(T &dst, const T &src, const V &vec) {
    static_assert(std::is_same<typename T::dtype, typename V::dtype>::value, "Tile and vector must have the same data type");
    static_assert(V::length == T::cols, "Vector length must match the number of columns in the tile");
#pragma unroll
    for(int i = laneid(); i < dst.num_elements; i += GROUP_THREADS) {
        int row = i/dst.cols, col = i%dst.cols;
        dst[{row, col}] = op::template op<typename T::dtype>(src[{row, col}], vec[col]);
    }
}


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// All of the annoying qualifiers *should* be automatically inferred during compile-time.
// So, syntax should just be kittens::add_row(tile, colvec);

// const maps
/**
 * @brief Sets all elements of the destination tile to zero.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void zero(T &dst) {
    unary_map<base_ops::zero, T>(dst, dst);
}
/**
 * @brief Sets all elements of the destination tile to one.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void one(T &dst) {
    unary_map<base_ops::one, T>(dst, dst);
}
/**
 * @brief Sets all elements of the destination tile to positive infinity.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void pos_infty(T &dst) {
    unary_map<base_ops::pos_infty, T>(dst, dst);
}
/**
 * @brief Sets all elements of the destination tile to negative infinity.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void neg_infty(T &dst) {
    unary_map<base_ops::neg_infty, T>(dst, dst);
}

// unary maps
/**
 * @brief Applies the exponential function to each element of the source tile and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the exponential function is applied.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void exp(T &dst, const T &src) {
    unary_map<base_ops::exp, T>(dst, src);
}
/**
 * @brief Applies the exponential function to each element of the source tile and stores the result in the destination tile, in base 2.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the exponential function is applied.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void exp2(T &dst, const T &src) {
    unary_map<base_ops::exp2, T>(dst, src);
}
/**
 * @brief Applies the natural logarithm function to each element of the source tile and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the natural logarithm function is applied.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void log(T &dst, const T &src) {
    unary_map<base_ops::log, T>(dst, src);
}
/**
 * @brief Applies the logarithm base 2 function to each element of the source tile and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the logarithm base 2 function is applied.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void log2(T &dst, const T &src) {
    unary_map<base_ops::log2, T>(dst, src);
}
/**
 * @brief Applies the absolute function to each element of the source tile and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the absolute function is applied.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void abs(T &dst, const T &src) {
    unary_map<base_ops::abs, T>(dst, src);
}
/**
 * @brief Applies the rectified linear unit function to each element of the source tile and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the rectified linear unit function is applied.
 */
template<ducks::st::all T>
__attribute__((device)) static inline void relu(T &dst, const T &src) {
    unary_map<base_ops::relu, T>(dst, src);
}
/**
 * @brief Copies the elements of the source tile to the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source data to be copied.
 */
template<ducks::st::all T, typename U>
__attribute__((device)) static inline void copy(T &dst, const U &src) {
    bin_map<base_ops::copy, T>(dst, src);
}

// uniform binary maps
/**
 * @brief Finds the maximum of each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__attribute__((device)) static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::max, T>(dst, lhs, rhs);
}
/**
 * @brief Finds the minimum of each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__attribute__((device)) static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::min, T>(dst, lhs, rhs);
}
/**
 * @brief Adds each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__attribute__((device)) static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sum, T>(dst, lhs, rhs);
}
/**
 * @brief Subtracts each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__attribute__((device)) static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sub, T>(dst, lhs, rhs);
}
/**
 * @brief Multiplies each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__attribute__((device)) static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::mul, T>(dst, lhs, rhs);
}
/**
 * @brief Divides each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 * 
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<ducks::st::all T, typename U>
__attribute__((device)) static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::div, T>(dst, lhs, rhs);
}

// Row and col maps

/**
 * @brief Adds row values to each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the addition on.
 * @param row_values[in] Column vector containing values to add to each row.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void add_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sum, T, V>(dst, src, row_values);
}
/**
 * @brief Subtracts row values from each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the subtraction on.
 * @param row_values[in] Column vector containing values to subtract from each row.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void sub_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sub, T, V>(dst, src, row_values);
}
/**
 * @brief Multiplies each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the multiplication on.
 * @param row_values[in] Column vector containing values to multiply each row by.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void mul_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::mul, T, V>(dst, src, row_values);
}
/**
 * @brief Divides each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the division on.
 * @param row_values[in] Column vector containing values to divide each row by.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void div_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::div, T, V>(dst, src, row_values);
}
/**
 * @brief Broadcast a vector into into a tile's rows.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param row_values[in] Column vector containing values to broadcast into rows.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void broadcast_row(T &dst, const V &row_values) {
    row_map<base_ops::copy2, T, V>(dst, dst, row_values);
}


// col maps
/**
 * @brief Adds column values to each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the addition on.
 * @param col_values[in] Row vector containing values to add to each column.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void add_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sum, T, V>(dst, src, col_values);
}
/**
 * @brief Subtracts column values from each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the subtraction on.
 * @param col_values[in] Row vector containing values to subtract from each column.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void sub_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::sub, T, V>(dst, src, col_values);
}
/**
 * @brief Multiplies each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the multiplication on.
 * @param col_values[in] Row vector containing values to multiply each column by.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void mul_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::mul, T, V>(dst, src, col_values);
}
/**
 * @brief Divides each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the division on.
 * @param col_values[in] Row vector containing values to divide each column by.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void div_col(T &dst, const T &src, const V &col_values) {
    col_map<base_ops::div, T, V>(dst, src, col_values);
}
/**
 * @brief Broadcast a vector into into a tile's columns.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param row_values[in] Row vector containing values to broadcast into cols.
 */
template<ducks::st::all T, ducks::sv::all V>
__attribute__((device)) static inline void broadcast_col(T &dst, const V &col_values) {
    col_map<base_ops::copy2, T, V>(dst, dst, col_values);
}
# 8 "/root/HipKittens//include/ops/group/shared/tile/tile.cuh" 2
# 1 "/root/HipKittens//include/ops/group/shared/tile/reductions.cuh" 1
/**
 * @file
 * @brief Group reductions on shared tiles.
 */

/**
 * Performs row-wise reduction on a matrix using a specified operation.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type with row layout.
 * @param row_accum The accumulator where the result of the reduction is stored.
 * @param src The source matrix on which to perform the reduction.
 * @param src_accum The initial value of the accumulator, used when reset is false.
 * @param reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 */
template<typename op, ducks::sv::all V, ducks::st::all T, bool reset>
__attribute__((device)) static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    using dtype = typename V::dtype;
    for (int row = laneid(); row < src.rows; row += GROUP_THREADS) {
        dtype accum = src[{row, 0}];
#pragma unroll
        for (int col = 1; col < src.cols; col++) {
            accum = op::template op<dtype>(accum, src[{row, col}]);
        }
        if (reset) {
            row_accum[row] = accum;
        } else {
            row_accum[row] = op::template op<dtype>(src_accum[row], accum);
        }
    }
}

/**
 * Performs column-wise reduction on a matrix using a specified operation.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The shared vector type for the column accumulator.
 * @tparam T The shared matrix type with column layout.
 * @param col_accum The accumulator where the result of the reduction is stored.
 * @param src The source matrix on which to perform the reduction.
 * @param src_accum The initial value of the accumulator, used when reset is false.
 * @param reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 */
template<typename op, ducks::sv::all V, ducks::st::all T, bool reset>
__attribute__((device)) static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    using dtype = typename V::dtype;
    for (int col = laneid(); col < src.cols; col += GROUP_THREADS) {
        dtype accum = src[{0, col}];
#pragma unroll
        for (int row = 1; row < src.rows; row++) {
            accum = op::template op<dtype>(accum, src[{row, col}]);
        }
        if (reset) {
            col_accum[col] = accum;
        } else {
            col_accum[col] = op::template op<dtype>(src_accum[col], accum);
        }
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

/**
 * @brief Store the maximum of each row of the src shared matrix in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void row_max(V &row_accum, const T &src) {
    row_reduce<base_ops::max, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the minimum of each row of the src shared matrix in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void row_min(V &row_accum, const T &src) {
    row_reduce<base_ops::min, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the sum of each row of the src shared matrix in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void row_sum(V &row_accum, const T &src) {
    row_reduce<base_ops::sum, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the product of each row of the src shared matrix in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void row_prod(V &row_accum, const T &src) {
    row_reduce<base_ops::mul, V, T, true>(row_accum, src, row_accum);
}

/**
 * @brief Store the maximum of each row of the src shared matrix, as well as the src_accum shared vector, in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void row_max(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::max, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the minimum of each row of the src shared matrix, as well as the src_accum shared vector, in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void row_min(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::min, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the sum of each row of the src shared matrix, as well as the src_accum shared vector, in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void row_sum(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::sum, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the product of each row of the src shared matrix, as well as the src_accum shared vector, in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void row_prod(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::mul, V, T, false>(row_accum, src, src_accum);
}

/**
 * @brief Store the maximum of each column of the src shared matrix in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void col_max(V &col_accum, const T &src) {
    col_reduce<base_ops::max, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the minimum of each column of the src shared matrix in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void col_min(V &col_accum, const T &src) {
    col_reduce<base_ops::min, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the sum of each column of the src shared matrix in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void col_sum(V &col_accum, const T &src) {
    col_reduce<base_ops::sum, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the product of each column of the src shared matrix in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void col_prod(V &col_accum, const T &src) {
    col_reduce<base_ops::mul, V, T, true>(col_accum, src, col_accum);
}

/**
 * @brief Store the maximum of each column of the src shared matrix, as well as the src_accum shared vector, in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void col_max(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::max, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the minimum of each column of the src shared matrix, as well as the src_accum shared vector, in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void col_min(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::min, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the sum of each column of the src shared tile, as well as the src_accum row vector, in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void col_sum(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::sum, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the product of each column of the src shared tile, as well as the src_accum row vector, in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__attribute__((device)) static inline void col_prod(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::mul, V, T, false>(col_accum, src, src_accum);
}
# 8 "/root/HipKittens//include/ops/group/shared/tile/tile.cuh" 2
# 7 "/root/HipKittens//include/ops/group/shared/shared.cuh" 2
# 1 "/root/HipKittens//include/ops/group/shared/vec/vec.cuh" 1
/**
 * @file
 * @brief An aggregate header for group operations on shared vectors.
 */

# 1 "/root/HipKittens//include/ops/group/shared/vec/conversions.cuh" 1
/**
 * @file
 * @brief Group conversions on shared vectors.
 */

/**
 * @brief Copies data from one shared vector to another, converting data types if necessary.
 *
 * This function copies data from the source shared vector `src` to the destination shared vector `dst`.
 * If the data types of `src` and `dst` are the same, it performs a direct memory copy. Otherwise, it
 * converts each element from the source data type to the destination data type using the appropriate
 * converter before copying.
 *
 * @tparam SV1 The type of the destination shared vector, must satisfy the ducks::sv::all concept.
 * @tparam SV2 The type of the source shared vector, must satisfy the ducks::sv::all concept.
 * @param[out] dst The destination shared vector.
 * @param[in] src The source shared vector.
 * @note The lengths of `src` and `dst` must be equal. This is enforced at compile time.
 */
template<ducks::sv::all SV1, ducks::sv::all SV2>
__attribute__((device)) static inline void copy(SV1 &dst, const SV2 &src) {
    static_assert(dst.length == src.length, "Source and destination vectors must have the same length.");
#pragma unroll
    for(int i = laneid(); i < dst.length; i+=GROUP_THREADS) {
        dst[i] = base_types::convertor<typename SV1::dtype, typename SV2::dtype>::convert(src[i]);
    }
}
# 7 "/root/HipKittens//include/ops/group/shared/vec/vec.cuh" 2
# 1 "/root/HipKittens//include/ops/group/shared/vec/maps.cuh" 1
/**
 * @file
 * @brief Group maps on shared vectors.
 */

/**
 * @brief Applies a unary operation to each element of a shared memory vector.
 *
 * @tparam op Unary operation type.
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector in which to store the result.
 * @param src[in] Source vector to apply the unary operation.
 */
template<typename op, ducks::sv::all T>
__attribute__((device)) static inline void unary_op(T &dst, const T &src) {
#pragma unroll
    for(auto cur = laneid(); cur < T::length; cur+=GROUP_THREADS) {
        dst[cur] = op::template op<typename T::dtype>(src[cur]);
    }
}
/**
 * @brief Perform a binary operation on two shared vectors.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vectors.
 * @param dst[out] The destination vector where the result is stored.
 * @param lhs[in] The left-hand side vector for the operation.
 * @param rhs[in] The right-hand side vector for the operation.
 */
template<typename op, ducks::sv::all T>
__attribute__((device)) static inline void bin_op(T &dst, const T &lhs, const T &rhs) {
#pragma unroll
    for(auto cur = laneid(); cur < T::length; cur+=GROUP_THREADS) {
        dst[cur] = op::template op<typename T::dtype>(lhs[cur], rhs[cur]);
    }
}
/**
 * @brief Perform a binary operation on a shared vector and a scalar.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vector.
 * @param dst[out] The destination vector where the result is stored.
 * @param src[in] The source vector for the operation.
 * @param param[in] The scalar parameter for the operation.
 */
template<typename op, ducks::sv::all T>
__attribute__((device)) static inline void bin_op(T &dst, const T &src, const typename T::dtype &param) {
#pragma unroll
    for(auto cur = laneid(); cur < T::length; cur+=GROUP_THREADS) {
        dst[cur] = op::template op<typename T::dtype>(src[cur], param);
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// ---- const ops ----

/**
 * @brief Sets all elements of a shared memory vector to zero.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to zero.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void zero(T &dst) {
    unary_op<base_ops::zero, T>(dst, dst);
}
/**
 * @brief Sets all elements of a shared memory vector to one.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to one.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void one(T &dst) {
    unary_op<base_ops::one, T>(dst, dst);
}
/**
 * @brief Sets all elements of a shared memory vector to positive infinity.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to positive infinity.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void pos_infty(T &dst) {
    unary_op<base_ops::pos_infty, T>(dst, dst);
}
/**
 * @brief Sets all elements of a shared memory vector to negative infinity.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to negative infinity.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void neg_infty(T &dst) {
    unary_op<base_ops::neg_infty, T>(dst, dst);
}

// ---- unary ops ----

/**
 * @brief Copies the elements from one shared vector to another.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the source vector.
 * @param dst[out] Destination vector where the elements will be copied to.
 * @param src[in] Source vector to copy the elements from.
 */
template<ducks::sv::all T, typename U>
__attribute__((device)) static inline void copy(T &dst, const U &src) {
    bin_op<base_ops::copy2, T>(dst, dst, src); // the second arg is ignored here.
}
/**
 * @brief Applies the exponential function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void exp(T &dst, const T &src) {
    unary_op<base_ops::exp, T>(dst, src);
}
/**
 * @brief Applies the exponential function element-wise to a shared vector, in base 2.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void exp2(T &dst, const T &src) {
    unary_op<base_ops::exp2, T>(dst, src);
}
/**
 * @brief Applies the natural logarithm function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the logarithm values will be stored.
 * @param src[in] Source vector to apply the logarithm function to.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void log(T &dst, const T &src) {
    unary_op<base_ops::log, T>(dst, src);
}
/**
 * @brief Applies the logarithm base 2 function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the logarithm base 2 values will be stored.
 * @param src[in] Source vector to apply the logarithm base 2 function to.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void log2(T &dst, const T &src) {
    unary_op<base_ops::log2, T>(dst, src);
}
/**
 * @brief Applies the absolute value function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the absolute values will be stored.
 * @param src[in] Source vector to apply the absolute value function to.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void abs(T &dst, const T &src) {
    unary_op<base_ops::abs, T>(dst, src);
}
/**
 * @brief Applies the rectified linear unit (ReLU) function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the ReLU values will be stored.
 * @param src[in] Source vector to apply the ReLU function to.
 */
template<ducks::sv::all T>
__attribute__((device)) static inline void relu(T &dst, const T &src) {
    unary_op<base_ops::relu, T>(dst, src);
}

// ---- binary ops ----

/**
 * @brief Computes the element-wise maximum of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the maximum values will be stored.
 * @param lhs[in] First vector for the maximum operation.
 * @param rhs[in] Second vector for the maximum operation.
 */
template<ducks::sv::all T, typename U>
__attribute__((device)) static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::max, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise minimum of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the minimum values will be stored.
 * @param lhs[in] First vector for the minimum operation.
 * @param rhs[in] Second vector for the minimum operation.
 */
template<ducks::sv::all T, typename U>
__attribute__((device)) static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::min, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise sum of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the sum values will be stored.
 * @param lhs[in] First vector for the sum operation.
 * @param rhs[in] Second vector for the sum operation.
 */
template<ducks::sv::all T, typename U>
__attribute__((device)) static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sum, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise difference of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the difference values will be stored.
 * @param lhs[in] First vector for the difference operation.
 * @param rhs[in] Second vector for the difference operation.
 */
template<ducks::sv::all T, typename U>
__attribute__((device)) static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sub, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise product of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the product values will be stored.
 * @param lhs[in] First vector for the product operation.
 * @param rhs[in] Second vector for the product operation.
 */
template<ducks::sv::all T, typename U>
__attribute__((device)) static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::mul, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise division of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the division values will be stored.
 * @param lhs[in] First vector for the division operation.
 * @param rhs[in] Second vector for the division operation.
 */
template<ducks::sv::all T, typename U>
__attribute__((device)) static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::div, T>(dst, lhs, rhs);
}
# 8 "/root/HipKittens//include/ops/group/shared/vec/vec.cuh" 2
// no group vector reductions as they would require additional shared memory and synchronization, and those side effects just aren't worth it.
// warp vector reductions should be plenty fast in 99.9% of situations.
# 7 "/root/HipKittens//include/ops/group/shared/shared.cuh" 2
# 29 "/root/HipKittens//include/ops/group/group.cuh" 2
};

using warpgroup = group<4>; // special scope commonly used by SM_90 and later.

}
# 9 "/root/HipKittens//include/ops/ops.cuh" 2
# 11 "/root/HipKittens//include/kittens.cuh" 2
# 1 "/root/HipKittens//include/pyutils/util.cuh" 1






template <typename T>
void check(T err, char const* const func, char const* const file,
           int const line)
{
    if (err != hipSuccess)
    {
        std::cerr << "HIP Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << hipGetErrorString(err) << " " << func << std::endl;
        //std::exit(EXIT_FAILURE);
    }
}
# 12 "/root/HipKittens//include/kittens.cuh" 2


// #include "pyutils/pyutils.cuh" // for simple binding without including torch
# 18 "256_256_64_16.cpp" 2
# 1 "/root/HipKittens//include/pyutils/pyutils.cuh" 1





namespace kittens {
namespace py {

template<typename T> struct from_object {
    static T make(pybind11::object obj) {
        return obj.cast<T>();
    }
};
template<ducks::gl::all GL> struct from_object<GL> {
    static GL make(pybind11::object obj) {
        // Check if argument is a torch.Tensor
        if (pybind11::hasattr(obj, "__class__") &&
            obj.attr("__class__").attr("__name__").cast<std::string>() == "Tensor") {

            // Check if tensor is contiguous
            if (!obj.attr("is_contiguous")().cast<bool>()) {
                throw std::runtime_error("Tensor must be contiguous");
            }
            if (obj.attr("device").attr("type").cast<std::string>() == "cpu") {
                throw std::runtime_error("Tensor must be on CUDA device");
            }

            // Get shape, pad with 1s if needed
            std::array<int, 4> shape = {1, 1, 1, 1};
            auto py_shape = obj.attr("shape").cast<pybind11::tuple>();
            size_t dims = py_shape.size();
            if (dims > 4) {
                throw std::runtime_error("Expected Tensor.ndim <= 4");
            }
            for (size_t i = 0; i < dims; ++i) {
                shape[4 - dims + i] = pybind11::cast<int>(py_shape[i]);
            }

            // Get data pointer using data_ptr()
            uint64_t data_ptr = obj.attr("data_ptr")().cast<uint64_t>();

            // Create GL object using make_gl
            return make_gl<GL>(data_ptr, shape[0], shape[1], shape[2], shape[3]);
        }
        throw std::runtime_error("Expected a torch.Tensor");
    }
};

template<typename T> concept has_dynamic_shared_memory = requires(T t) { { t.dynamic_shared_memory() } -> std::convertible_to<int>; };

template<typename> struct trait;
template<typename MT, typename T> struct trait<MT T::*> { using member_type = MT; using type = T; };
template<typename> using object = pybind11::object;
template<auto kernel, typename TGlobal> static void bind_kernel(auto m, auto name, auto TGlobal::*... member_ptrs) {
    m.def(name, [](object<decltype(member_ptrs)>... args) {
        TGlobal __g__ {from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...};
        if constexpr (has_dynamic_shared_memory<TGlobal>) {
            int __dynamic_shared_memory__ = (int)__g__.dynamic_shared_memory();
            hipFuncSetAttribute((void *) kernel, hipFuncAttributeMaxDynamicSharedMemorySize, __dynamic_shared_memory__);
            kernel<<<__g__.grid(), __g__.block(), __dynamic_shared_memory__>>>(__g__);
        } else {
            kernel<<<__g__.grid(), __g__.block()>>>(__g__);
        }
    });
}
template<auto function, typename TGlobal> static void bind_function(auto m, auto name, auto TGlobal::*... member_ptrs) {
    m.def(name, [](object<decltype(member_ptrs)>... args) {
        TGlobal __g__ {from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...};
        function(__g__);
    });
}

} // namespace py
} // namespace kittens
# 19 "256_256_64_16.cpp" 2
using namespace kittens;

constexpr int BLOCK_SIZE = 256;
constexpr int K_STEP = 64;
constexpr int REG_BLOCK = BLOCK_SIZE / 4;
constexpr int DOT_SLICE = 16;
# 33 "256_256_64_16.cpp"
using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<8>;


struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    hipStream_t stream;
    dim3 grid() { return dim3((8192 / BLOCK_SIZE) * (8192 / BLOCK_SIZE)); }
    dim3 block() { return dim3((kittens::WARP_THREADS * 8)); }
    size_t dynamic_shared_memory() { return 65536; }
};

__attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1, (kittens::WARP_THREADS * 8)), amdgpu_waves_per_eu(2)))
void micro_tk(const micro_globals g) {
    extern __attribute__((shared)) alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, K_STEP> (&As) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>(); // BM=256, BN=256, BK=64
    st_bf<BLOCK_SIZE, K_STEP> (&Bs) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();

    rt_bf<REG_BLOCK, DOT_SLICE> tiles[8]; // REG_BLOCK=64, DOT_SLICE=16
    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[2]; // REG_BLOCK=64, REG_BLOCK=64, layout=col
    for (int i = 0; i < 2; i++) { zero(C_accum[i]); }
    // C_accum[0].height /*4*/ // 64 * 64 / 4 / 4 = 256.
    // C_accum[0].width /*4*/

    // Get original WGID.
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    constexpr int WGM = 4;
    // Swizzle chiplet so that wgids are in the same XCD.
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);
    // Swizzle for better L2 within the same XCD.
    const int num_pid = ceil_div(8192, BLOCK_SIZE);
    int num_wgid_in_group = WGM * num_pid;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    // Assign the tile's row/column based on the pid_m and pid_n.
    const int row = pid_m;
    const int col = pid_n;


    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;

    const int num_tiles = 8192 / K_STEP;

    // Load first tile into shared memory
    G::load(As, g.a, {0, 0, row, 0});
    G::load(Bs, g.b, {0, 0, col, 0});
    __builtin_amdgcn_s_barrier();
    if constexpr (false) {
        kittens::load<2,false>(As, g.a, {0, 0, row, 0});
        constexpr auto _256 = st_bf<BLOCK_SIZE, K_STEP>::rows;
        constexpr auto _64 = st_bf<BLOCK_SIZE, K_STEP>::cols;
        using _ST = st_bf<BLOCK_SIZE, K_STEP>;
        using _GL = gl<__hip_bfloat16, -1, -1, -1, -1>;
        using idx = coord<_ST>;
        auto idx_now = idx{0,0,row,0};
        coord<> unit_coord = idx_now.unit_coord<2, 3>();
        auto *src_ptr = &g.a[unit_coord];
    }


    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

#pragma unroll
    for (int tile = 0; tile < num_tiles - 1; ++tile) {

        // Small register buffers for pipelining
        // This is used instead of register tiles to enable the use of maximally coalesced global loads.
        // 256 * 64 / 512 = 32;  // 一个 tile.
        constexpr int BUFFER_SIZE = (BLOCK_SIZE * K_STEP) / (kittens::WARP_THREADS * 8);
        float4 a_buffer_next[BUFFER_SIZE
            * sizeof(bf16) / sizeof(float4)
        ];
        float4 b_buffer_next[BUFFER_SIZE
            * sizeof(bf16) / sizeof(float4)
        ];

        // Cluster 0
        load_global_to_register_buffer<2, false, (kittens::WARP_THREADS * 8)>(a_buffer_next, BUFFER_SIZE, g.a, {0, 0, row, tile + 1}, As);
        if(0){
            kittens::zz::load_global_to_register_buffer(a_buffer_next, BUFFER_SIZE, g.a, {0, 0, row, tile + 1}, As);
        }
        // 64 * 16 / 512 = 2; // 一个小的subtile 横着 4个, 竖着 4 个
        load(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 0}));
        if constexpr (false) {
            auto now = subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 0});
            zz2::load(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 0}));
        }
        load(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 0}));
        load(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 0}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
        mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
        if(0){
            using _D = rt<float, 64, 64, ducks::rt_layout::col>;
            using _A = rt<__hip_bfloat16, 64, 16, kittens::ducks::rt_layout::row>;
            using _B = rt<__hip_bfloat16, 64, 16, kittens::ducks::rt_layout::row>;
            auto __4 = _D::height;
            auto _4 = _D::width;
            auto __1 = _A::width;
            // __builtin_amdgcn_mfma_f32_16x16x16bf16_1k;
            // 16次 mfma 
        }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2
        load(tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 1}));
        load(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 1}));
        load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 1}));
        load(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 2}));
        load(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 2}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3
        // asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(8)"); // 2 * 4(signle load)
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
        mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4
        load_global_to_register_buffer<2, false, (kittens::WARP_THREADS * 8)>(b_buffer_next, BUFFER_SIZE, g.b, {0, 0, col, tile + 1}, Bs);
        load(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 2}));
        load(tiles[6], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 3}));
        load(tiles[7], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 3}));
        load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5
        asm volatile("s_waitcnt lgkmcnt(12)"); // 2 * 4(signle load)
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
        mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6
        // asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)"); // 3 * 4(signle load)
        store_register_buffer_to_shared<(kittens::WARP_THREADS * 8)>(As, a_buffer_next);
        store_register_buffer_to_shared<(kittens::WARP_THREADS * 8)>(Bs, b_buffer_next);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[7], tiles[6], C_accum[0]);
        mma_ABt(C_accum[1], tiles[5], tiles[6], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

    }

    // Epilogue
    // Cluster 0
    __builtin_amdgcn_sched_barrier(0);
    load(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 0}));
    load(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 0}));
    load(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);


    // Cluster 1
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
    mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2
    load(tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 1}));
    load(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 1}));
    load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 1}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 3
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
    mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 4
    load(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 2}));
    load(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 2}));
    load(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 2}));
    load(tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 3}));
    load(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 3}));
    load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 3}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 5
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
    mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 7
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
    mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }

    store(g.c, C_accum[0], {0, 0, row * 4 + warp_row, col * 4 + warp_col});
    store(g.c, C_accum[1], {0, 0, row * 4 + warp_row + 2, col * 4 + warp_col});
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

static int pybind11_exec_tk_kernel(PyObject *); extern "C" [[maybe_unused]] __attribute__((visibility("default"))) PyObject *PyInit_tk_kernel(); extern "C" __attribute__((visibility("default"))) PyObject *PyInit_tk_kernel() { { const char *compiled_ver = "3" "." "12"; const char *runtime_ver = Py_GetVersion(); size_t len = std::strlen(compiled_ver); if (std::strncmp(runtime_ver, compiled_ver, len) != 0 || (runtime_ver[len] >= '0' && runtime_ver[len] <= '9')) { PyErr_Format(PyExc_ImportError, "Python version mismatch: module was compiled for Python %s, " "but the interpreter version is incompatible: %s.", compiled_ver, runtime_ver); return nullptr; } } (pybind11::detail::get_num_interpreters_seen() += 1); { pybind11::detail::get_internals_pp_manager().unref(); pybind11::detail::get_internals(); } static ::pybind11::detail::slots_array mod_def_slots = ::pybind11::detail::init_slots( &pybind11_exec_tk_kernel); static PyModuleDef def{ { { { 1 }, (nullptr) }, nullptr, 0, nullptr, }, "tk_kernel", nullptr, 0, nullptr, mod_def_slots.data(), nullptr, nullptr, nullptr}; return PyModuleDef_Init(&def); } static void pybind11_init_tk_kernel(::pybind11::module_ &); int pybind11_exec_tk_kernel(PyObject * pm) { try { auto m = pybind11::reinterpret_borrow<::pybind11::module_>(pm); if (!pybind11::detail::get_cached_module(m.attr("__spec__").attr("name"))) { pybind11_init_tk_kernel(m); pybind11::detail::cache_completed_module(m); } return 0; } catch (pybind11::error_already_set & e) { pybind11::raise_from(e, PyExc_ImportError, "initialization failed"); } catch (const std::exception &e) { ::pybind11::set_error(PyExc_ImportError, e.what()); } return -1; } void pybind11_init_tk_kernel(::pybind11::module_ & m) {
    m.doc() = "tk_kernel python module";
    py::bind_kernel<micro_tk>(m, "micro_tk", &micro_globals::a, &micro_globals::b, &micro_globals::c);
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}

// __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
