#include <concepts>
#include <type_traits>

namespace ducks {
namespace rt_layout {
struct row {};
struct col {};

template <typename T>
concept all = std::is_same_v<T, row> || std::is_same_v<T, col>;
} // namespace rt_layout

namespace rt {
/**
 * @brief A dummy type used to identify register tiles.
 * 
 * For a type to quack like an rt, it should define its identifier as ducks::rt::identifier.
 * If a type quacks like ducks::rt::identifier, it will be treated as an rt by compiler checks.
 */ 
struct identifier {};
template<typename T> concept all = requires{
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;

template<typename T>
concept row_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::row>;

template<typename T> 
concept col_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::col>;
} // namespace rt


} // namespace ducks

template <ducks::rt::col_layout D, ducks::rt::row_layout A>
static inline void mma_ABt(D &d, const A &a) {
  static_assert(D::rows == A::rows);
}

constexpr int BLOCK_SIZE = 256;
constexpr int REG_BLOCK = BLOCK_SIZE / 4;
constexpr int DOT_SLICE = 16;

using bf16 = float;

template <typename _T, int _rows, int _cols, ducks::rt_layout::all _layout>
struct rt {
    using identifier = ducks::rt::identifier;
    using layout = _layout;
    static constexpr int rows = _rows;
    static constexpr int cols = _cols;
};
template <int _r, int _c, ducks::rt_layout::all layout = ducks::rt_layout::row>
using rt_bf = rt<bf16, _r, _c, layout>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row>
using rt_fl = rt<float, _r, _c, layout>; 

int main() {
  rt_bf<REG_BLOCK, DOT_SLICE> tiles[8]; // REG_BLOCK=64, DOT_SLICE=16
  rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[2];
  mma_ABt(C_accum[0], tiles[1]);
}