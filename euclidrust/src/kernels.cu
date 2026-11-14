typedef unsigned long long ull;
#define POS_INF __longlong_as_double(0x7FF0000000000000LL)
#define NEG_INF __longlong_as_double((long long)0xFFF0000000000000ULL)

template <int BP> __device__ __forceinline__ double load(const unsigned char* d, ull i);
template <> __device__ __forceinline__ double load<8>(const unsigned char* d, ull i) {
    return (double)d[i];
}
template <> __device__ __forceinline__ double load<16>(const unsigned char* d, ull i) {
    unsigned short u = ((const unsigned short*)d)[i];
    u = (unsigned short)((u << 8) | (u >> 8));
    return (double)(short)u;
}
template <> __device__ __forceinline__ double load<32>(const unsigned char* d, ull i) {
    unsigned int u = ((const unsigned int*)d)[i];
    u = __byte_perm(u, 0, 0x0123);
    return (double)(int)u;
}
template <> __device__ __forceinline__ double load<-32>(const unsigned char* d, ull i) {
    unsigned int u = ((const unsigned int*)d)[i];
    u = __byte_perm(u, 0, 0x0123);
    return (double)__uint_as_float(u);
}
template <> __device__ __forceinline__ double load<-64>(const unsigned char* d, ull i) {
    ull u = ((const ull*)d)[i];
    unsigned int lo = __byte_perm((unsigned int)(u & 0xFFFFFFFFULL), 0, 0x0123);
    unsigned int hi = __byte_perm((unsigned int)(u >> 32), 0, 0x0123);
    return __longlong_as_double((long long)(((ull)lo << 32) | (ull)hi));
}

template <int BP>
__device__ __forceinline__ void reduce(const unsigned char* d, ull n, double bzero, double bscale, double* out) {
    double mn = POS_INF, mx = NEG_INF, sum = 0.0;
    ull nans = 0;
    ull stride = (ull)gridDim.x * blockDim.x;
    for (ull i = (ull)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        double v = bzero + bscale * load<BP>(d, i);
        if (v != v) { nans++; continue; }
        mn = fmin(mn, v);
        mx = fmax(mx, v);
        sum += v;
    }
    for (int o = 16; o > 0; o >>= 1) {
        mn = fmin(mn, __shfl_down_sync(0xffffffffu, mn, o));
        mx = fmax(mx, __shfl_down_sync(0xffffffffu, mx, o));
        sum += __shfl_down_sync(0xffffffffu, sum, o);
        nans += __shfl_down_sync(0xffffffffu, nans, o);
    }
    __shared__ double smn[32], smx[32], ssum[32];
    __shared__ ull snan[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { smn[warp] = mn; smx[warp] = mx; ssum[warp] = sum; snan[warp] = nans; }
    __syncthreads();
    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;
        mn = lane < nw ? smn[lane] : POS_INF;
        mx = lane < nw ? smx[lane] : NEG_INF;
        sum = lane < nw ? ssum[lane] : 0.0;
        nans = lane < nw ? snan[lane] : 0;
        for (int o = 16; o > 0; o >>= 1) {
            mn = fmin(mn, __shfl_down_sync(0xffffffffu, mn, o));
            mx = fmax(mx, __shfl_down_sync(0xffffffffu, mx, o));
            sum += __shfl_down_sync(0xffffffffu, sum, o);
            nans += __shfl_down_sync(0xffffffffu, nans, o);
        }
        if (lane == 0) {
            out[blockIdx.x * 4 + 0] = mn;
            out[blockIdx.x * 4 + 1] = mx;
            out[blockIdx.x * 4 + 2] = sum;
            out[blockIdx.x * 4 + 3] = (double)nans;
        }
    }
}

#define N_RANDOM 10000
#define ZERO_VALUE (-2147483646LL)

// Decodes one Rice-compressed tile and accumulates its statistics without writing the pixels
// out. This is the algorithm of cfitsio's fits_rdecomp (BP = bytes per integer: 4, 2 or 1),
// followed by the same de-quantization and subtractive dither as the CPU code.
#define EMIT(UV) { \
    long long vi = (BP == 4) ? (long long)(int)(unsigned int)(UV) \
                 : ((BP == 2) ? (long long)(short)(unsigned short)(UV) : (long long)(unsigned char)(UV)); \
    double val; \
    if (is_float) { \
        if (vi == tnull) val = __longlong_as_double(0x7FF8000000000000LL); \
        else if (dither == 2 && vi == ZERO_VALUE) val = 0.0; \
        else if (dither == 0) val = (double)vi * scale + zero; \
        else val = ((double)vi - (double)rnd[nextrand] + 0.5) * scale + zero; \
        if (f32_out) val = (double)(float)val; \
        if (dither) { \
            nextrand++; \
            if (nextrand == N_RANDOM) { iseed++; if (iseed == N_RANDOM) iseed = 0; nextrand = (int)(rnd[iseed] * 500.0f); } \
        } \
    } else { val = bzero + bscale * (double)vi; } \
    if (val != val) nans++; else { mn = fmin(mn, val); mx = fmax(mx, val); sum += val; } }

template <int BP>
__device__ __forceinline__ void rice_stats(const unsigned char* c, unsigned int clen, unsigned int n, int nblock,
        double scale, double zero, long long tnull, int dither, int iseed, int is_float, int f32_out,
        double bzero, double bscale, const float* rnd, double* out) {
    const int fsbits = (BP == 4) ? 5 : ((BP == 2) ? 4 : 3);
    const int fsmax = (BP == 4) ? 25 : ((BP == 2) ? 14 : 6);
    const int bbits = BP * 8;
    const ull mask = (BP == 4) ? 0xFFFFFFFFULL : ((BP == 2) ? 0xFFFFULL : 0xFFULL);
    unsigned int pos = 0;
    bool bad = false;
#define NEXT() ((pos < clen) ? (ull)c[pos++] : (bad = true, (ull)0))
    ull lastpix = 0;
    for (int k = 0; k < BP; k++) lastpix = (lastpix << 8) | NEXT();
    ull b = NEXT();
    int nbits = 8;
    double mn = POS_INF, mx = NEG_INF, sum = 0.0;
    ull nans = 0;
    int nextrand = dither ? (int)(rnd[iseed] * 500.0f) : 0;
    unsigned int i = 0;
    while (i < n && !bad) {
        nbits -= fsbits;
        while (nbits < 0) { b = (b << 8) | NEXT(); nbits += 8; }
        int fs = (int)(b >> nbits) - 1;
        b &= (1ULL << nbits) - 1;
        unsigned int imax = i + (unsigned int)nblock;
        if (imax > n) imax = n;
        if (fs < 0) {
            for (; i < imax; i++) EMIT(lastpix)
        } else if (fs == fsmax) {
            for (; i < imax; i++) {
                int k = bbits - nbits;
                ull diff = b << k;
                k -= 8;
                while (k >= 0) { b = NEXT(); diff |= b << k; k -= 8; }
                if (nbits > 0) { b = NEXT(); diff |= b >> (-k); b &= (1ULL << nbits) - 1; } else { b = 0; }
                diff &= mask;
                ull d = ((diff & 1) == 0) ? (diff >> 1) : ~(diff >> 1);
                lastpix = (d + lastpix) & mask;
                EMIT(lastpix)
            }
        } else {
            for (; i < imax; i++) {
                while (b == 0) { nbits += 8; b = NEXT(); if (bad) break; }
                if (bad) break;
                int nzero = nbits - (64 - __clzll((long long)b));
                nbits -= nzero + 1;
                b ^= 1ULL << nbits;
                nbits -= fs;
                while (nbits < 0) { b = (b << 8) | NEXT(); nbits += 8; }
                ull diff = ((ull)nzero << fs) | (b >> nbits);
                b &= (1ULL << nbits) - 1;
                ull d = ((diff & 1) == 0) ? (diff >> 1) : ~(diff >> 1);
                lastpix = (d + lastpix) & mask;
                EMIT(lastpix)
            }
        }
    }
    out[0] = mn; out[1] = mx; out[2] = sum;
    out[3] = bad ? 1e30 : (double)nans;   // 1e30 marks a corrupt tile
}

extern "C" __global__ void rice_kernel(const unsigned char* heap, const ull* off, const unsigned int* clen,
        const unsigned int* npix, const double* scale, const double* zero, const long long* tnull,
        int n_tiles, int nblock, int bytepix, int dither, long long dither_seed, int is_float, int f32_out,
        double bzero, double bscale, const float* rnd, double* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tiles) return;
    int iseed = (int)((((long long)t + dither_seed - 1) % N_RANDOM + N_RANDOM) % N_RANDOM);
    const unsigned char* c = heap + off[t];
    switch (bytepix) {
        case 4: rice_stats<4>(c, clen[t], npix[t], nblock, scale[t], zero[t], tnull[t], dither, iseed, is_float, f32_out, bzero, bscale, rnd, out + 4 * (size_t)t); break;
        case 2: rice_stats<2>(c, clen[t], npix[t], nblock, scale[t], zero[t], tnull[t], dither, iseed, is_float, f32_out, bzero, bscale, rnd, out + 4 * (size_t)t); break;
        case 1: rice_stats<1>(c, clen[t], npix[t], nblock, scale[t], zero[t], tnull[t], dither, iseed, is_float, f32_out, bzero, bscale, rnd, out + 4 * (size_t)t); break;
    }
}

extern "C" __global__ void stats_kernel(const unsigned char* d, ull n, int bitpix, double bzero, double bscale, double* out) {
    switch (bitpix) {
        case 8:   reduce<8>(d, n, bzero, bscale, out); break;
        case 16:  reduce<16>(d, n, bzero, bscale, out); break;
        case 32:  reduce<32>(d, n, bzero, bscale, out); break;
        case -32: reduce<-32>(d, n, bzero, bscale, out); break;
        case -64: reduce<-64>(d, n, bzero, bscale, out); break;
    }
}
