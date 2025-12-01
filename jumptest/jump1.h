void mma_AB_base(int* a){ }
void mma_AB_base(float* a){ }
void mma_AB_base(double* a){ }


template<typename T> 
struct rt{
    static constexpr int rows = sizeof(T);
    T tiles[rows];
};