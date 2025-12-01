template<typename T>
static inline void mma_AB(T a){ 
    const auto now = a.rows;
    mma_AB_base(a.tiles);
}