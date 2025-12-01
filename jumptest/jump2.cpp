void mma_AB_base(int* a){ }
void mma_AB_base(float* a){ }
void mma_AB_base(double* a){ }

template<typename T> 
struct rt{
    static constexpr int rows = sizeof(T);
    T tiles[rows];
};

template<typename T>
static inline void mma_AB(T a){ 
    const auto now = a.rows;
    mma_AB_base(a.tiles);
}

int main(){
    rt<float> a;
    mma_AB(a);
    rt<double> b; 
    mma_AB(b);
}
