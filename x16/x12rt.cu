/**
 * X12R algorithm (X12 with Randomized chain order)
 *
 *  Soteria 2025
 */
// x12rt.cu — x12 (soterg) implementation
// - deterministic nibble selection (GetHashSelection)
// - TIME_MASK 0xFFFFFFA0
// - getprevblock hashes big-endian masked timestamp
// - getAlgoString uses direct assignment
// - sph contexts scoped per-case to reduce stack usage
// - canonical timestamp normalization (swab32)
// - safer cudaMalloc/init ordering and basic bounds checks

#include <stdio.h>
#include <memory.h>
#include <unistd.h>
#include <assert.h>
#include <arpa/inet.h> // htonl

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_shabal.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "sph/sph_fugue.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_sha2.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
#include "sph/sph_hamsi.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x16.h"

// --- Constants and globals -------------------------------------------------

#ifndef HASH_FUNC_COUNT
#define HASH_FUNC_COUNT 12
#endif

// x12 specific TIME_MASK
#define TIME_MASK 0xFFFFFFA0U

static uint32_t *d_hash[MAX_GPUS];
static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };
static uint32_t thr_throughput[MAX_GPUS] = { 0 };
static int algo80_fails[HASH_FUNC_COUNT] = { 0 };

enum Algo {
    BLAKE = 0,
    SHABAL,
    GROESTL,
    JH,
    KECCAK,
    SKEIN,
    FUGUE,
    CUBEHASH,
    SIMD,
    ECHO,
    HAMSI,
    SHA512,
    HASH_FUNC_COUNT_ENUM
};

static const char* algo_strings[] = {
    "blake","shabal","groestl","jh512","keccak","skein","fugue","cube",
    "simd","echo","hamsi","sha512", NULL
};

// Thread-local cache of last timestamp-derived order
static __thread uint32_t s_ntime = UINT32_MAX;
static uint8_t s_firstalgo = 0xFF;
static __thread char hashOrder[HASH_FUNC_COUNT + 1] = { 0 };

// Forward declarations
static void init_x12rt(const int thr_id, int dev_id);

// --- PrevBlock nibble helper (byte-array based) ----------------------------
// Provide GetNibble and GetHashSelection equivalent for the provided byte array.
// PrevBlockHash is 32 bytes passed as uint32_t[8] or uint8_t*.

static inline uint8_t GetNibbleFromPos(const uint8_t* data, int pos)
{
    // pos in 0..63, 0 is leftmost nibble index matching original mapping
    int pos_rev = 63 - pos;
    int byte_idx = pos_rev >> 1;
    uint8_t b = data[byte_idx];
    if (pos_rev & 1)
        return b & 0x0F;        // low nibble
    return (b >> 4) & 0x0F;     // high nibble
}

static inline int GetHashSelection(const uint32_t* prevblock, int index)
{
    const uint8_t* data = (const uint8_t*)prevblock;
    const int START = 48;
    const int MASK  = 0xF;
    assert(index >= 0 && index < 12);
    int pos = START + (index & MASK);
    int nibble = GetNibbleFromPos(data, pos);
    if (nibble < 12) return nibble;

    for (int i = 1; i < 16; ++i) {
        pos = START + ((index + i) & MASK);
        nibble = GetNibbleFromPos(data, pos);
        if (nibble < 12) return nibble;
    }
    return GetNibbleFromPos(data, pos) % 12;
}

// --- Build ASCII order string from prevblock -------------------------------
static void getAlgoString(const uint32_t* prevblock, char *output)
{
    char *sptr = output;
    const uint8_t* data = (const uint8_t*)prevblock;
    for (uint8_t j = 0; j < HASH_FUNC_COUNT; j++) {
        int hashSelection = GetHashSelection(prevblock, j);
        if (hashSelection >= 10)
            *sptr++ = 'A' + (hashSelection - 10);
        else
            *sptr++ = '0' + hashSelection;
    }
    *sptr = '\0';
}

// --- prevblock derivation with defined byte-order for sha256d input ----------
static void getprevblock(const uint32_t timeStamp, void* prevblock)
{
    uint32_t masked = timeStamp & TIME_MASK;
    uint32_t be_masked = htonl(masked);
    sha256d((unsigned char*)prevblock, (const unsigned char*)&be_masked, sizeof(be_masked));
}

// X12rt GPU hash (chained 12 algorithms) for Validation, contexts scoped per-case 
extern "C" void x12rt_hash(void *output, const void *input)
{
    unsigned char _ALIGN(64) hash[128];
    void *in = (void*) input;
    int size = 80;
    uint32_t *in32 = (uint32_t*)input;
    uint32_t ntime_be = in32[17];
    uint32_t ntime = swab32(ntime_be); // normalize to host-endian (match scanhash path)
    uint32_t _ALIGN(64) prevblock[8];
    getprevblock(ntime, &prevblock);
    getAlgoString(&prevblock[0], hashOrder);

    for (int i = 0; i < HASH_FUNC_COUNT; i++) {
        const char elem = hashOrder[i];
        const uint8_t algo = (elem >= 'A') ? (elem - 'A' + 10) : (elem - '0');

        switch (algo) {
        case BLAKE: {
            sph_blake512_context ctx;
            sph_blake512_init(&ctx);
            sph_blake512(&ctx, in, size);
            sph_blake512_close(&ctx, hash);
            break;
        }
        case SHABAL: {
            sph_shabal512_context ctx;
            sph_shabal512_init(&ctx);
            sph_shabal512(&ctx, in, size);
            sph_shabal512_close(&ctx, hash);
            break;
        }
        case GROESTL: {
            sph_groestl512_context ctx;
            sph_groestl512_init(&ctx);
            sph_groestl512(&ctx, in, size);
            sph_groestl512_close(&ctx, hash);
            break;
        }
        case JH: {
            sph_jh512_context ctx;
            sph_jh512_init(&ctx);
            sph_jh512(&ctx, in, size);
            sph_jh512_close(&ctx, hash);
            break;
        }
        case KECCAK: {
            sph_keccak512_context ctx;
            sph_keccak512_init(&ctx);
            sph_keccak512(&ctx, in, size);
            sph_keccak512_close(&ctx, hash);
            break;
        }
        case SKEIN: {
            sph_skein512_context ctx;
            sph_skein512_init(&ctx);
            sph_skein512(&ctx, in, size);
            sph_skein512_close(&ctx, hash);
            break;
        }
        case FUGUE: {
            sph_fugue512_context ctx;
            sph_fugue512_init(&ctx);
            sph_fugue512(&ctx, in, size);
            sph_fugue512_close(&ctx, hash);
            break;
        }
        case CUBEHASH: {
            sph_cubehash512_context ctx;
            sph_cubehash512_init(&ctx);
            sph_cubehash512(&ctx, in, size);
            sph_cubehash512_close(&ctx, hash);
            break;
        }
        case SIMD: {
            sph_simd512_context ctx;
            sph_simd512_init(&ctx);
            sph_simd512(&ctx, in, size);
            sph_simd512_close(&ctx, hash);
            break;
        }
        case ECHO: {
            sph_echo512_context ctx;
            sph_echo512_init(&ctx);
            sph_echo512(&ctx, in, size);
            sph_echo512_close(&ctx, hash);
            break;
        }
        case HAMSI: {
            sph_hamsi512_context ctx;
            sph_hamsi512_init(&ctx);
            sph_hamsi512(&ctx, in, size);
            sph_hamsi512_close(&ctx, hash);
            break;
        }
        case SHA512: {
            sph_sha512_context ctx;
            sph_sha512_init(&ctx);
            sph_sha512(&ctx, in, size);
            sph_sha512_close(&ctx, hash);
            break;
        }
        in = (void*) hash;
        size = 64;
    }
    memcpy(output, hash, 32);
}

// --- Initialization ---------------------------------------------------------
static void init_x12rt(const int thr_id, const int dev_id)
{
    if (thr_id < 0 || thr_id >= MAX_GPUS) return;
    if (dev_id < 0) return;
    if (!device_name[dev_id]) return;

    int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 20 : 19;
    if (strstr(device_name[dev_id], "GTX 1080")) intensity = 20;
    uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);

    // Set device and scheduling flags before heavy initialization
    cudaSetDevice(device_map[thr_id]);
    if (opt_cudaschedule == -1 && gpu_threads == 1) {
        cudaDeviceReset();
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    }

    gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads",
          throughput2intensity(throughput), throughput);
    thr_throughput[thr_id] = throughput;

    // Per-algo initializers (x12 subset)
    quark_blake512_cpu_init(thr_id, throughput);
    quark_groestl512_cpu_init(thr_id, throughput);
    quark_skein512_cpu_init(thr_id, throughput);
    quark_jh512_cpu_init(thr_id, throughput);
    quark_keccak512_cpu_init(thr_id, throughput);
    x11_simd512_cpu_init(thr_id, throughput);
    x13_hamsi512_cpu_init(thr_id, throughput);
    x16_echo512_cuda_init(thr_id, throughput);
    x11_echo512_cpu_init(thr_id, throughput);
    x13_fugue512_cpu_init(thr_id, throughput);
    x16_fugue512_cpu_init(thr_id, throughput);
    x14_shabal512_cpu_init(thr_id, throughput);
    x17_sha512_cpu_init(thr_id, throughput);
}

// --- scanhash (host-side main loop) -----------------------------------------
extern "C" int scanhash_x12rt(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
    if (thr_id < 0 || thr_id >= MAX_GPUS) return 0;
    if (!work) return 0;

    uint32_t *pdata = work->data;
    uint32_t *ptarget = work->target;
    const uint32_t first_nonce = pdata[19];
    const int dev_id = device_map[thr_id];

    // One-time per-thread init and allocation
    if (!init[thr_id]) {
        init_x12rt(thr_id, dev_id);
        size_t bytes = (size_t)64 * (size_t)thr_throughput[thr_id];
        if (bytes == 0) return 0;
        if (cudaMalloc(&d_hash[thr_id], bytes) != cudaSuccess) {
            gpulog(LOG_ERR, thr_id, "cudaMalloc failed for %zu bytes", bytes);
            return 0;
        }
        cuda_check_cpu_init(thr_id, thr_throughput[thr_id]);
        init[thr_id] = true;
    }

    uint32_t throughput = thr_throughput[thr_id];

    if (opt_benchmark) {
        ((uint32_t*)ptarget)[7] = 0x003f;
        ((uint32_t*)pdata)[1] = 0xEFCDAB89;
        ((uint32_t*)pdata)[2] = 0x67452301;
    }

    uint32_t _ALIGN(64) endiandata[20];
    for (int k = 0; k < 19; k++)
        be32enc(&endiandata[k], pdata[k]);

    uint32_t _ALIGN(64) prevblock[8];
    uint32_t ntime_be = pdata[17];
    uint32_t ntime = swab32(ntime_be); // canonical host-endian timestamp

    if (s_ntime != ntime) {
        getprevblock(ntime, &prevblock);
        getAlgoString(&prevblock[0], hashOrder);
        s_ntime = ntime;
        if (!thr_id)
            applog(LOG_INFO, "hash order: %s time: (%08x) time hash: (%08x)", hashOrder, ntime, prevblock[0]);
    }

    cuda_check_cpu_setTarget(ptarget);
    const int hashes = HASH_FUNC_COUNT;
    const char first = hashOrder[0];
    const uint8_t algo80 = (first >= 'A') ? (first - 'A' + 10) : (first - '0');

    if (algo80 != s_firstalgo) {
        s_firstalgo = algo80;
        gpulog(LOG_INFO, thr_id, CL_GRN "Algo is now %s, Order %s", algo_strings[algo80 % HASH_FUNC_COUNT], hashOrder);
    }
    // Set block for initial 80-byte round (x12 subset)
    switch (algo80) {
    case BLAKE:
        quark_blake512_cpu_setBlock_80(thr_id, endiandata);
        break;
   case SHABAL:
	x16_shabal512_setBlock_80((void*)endiandata);
	break;
    case GROESTL:
        groestl512_setBlock_80(thr_id, endiandata);
        break;
    case JH:
        jh512_setBlock_80(thr_id, endiandata);
        break;
    case KECCAK:
        keccak512_setBlock_80(thr_id, endiandata);
        break;
    case SKEIN:
        skein512_cpu_setBlock_80((void*)endiandata);
        break;
    case FUGUE:
	x16_fugue512_setBlock_80((void*)pdata);
	break;
    case CUBEHASH:
        cubehash512_setBlock_80(thr_id, endiandata);
        break;
    case SIMD:
        x16_simd512_setBlock_80((void*)endiandata);
        break;
    case ECHO:
        x16_echo512_setBlock_80((void*)endiandata);
        break;
    case HAMSI:
        x13_hamsi512_setBlock_80((void*)endiandata);
        break;
    case SHA512:
	x16_sha512_setBlock_80(endiandata);
	break;
    default:
        if (!thr_id)
            applog(LOG_WARNING, "kernel %s %c unimplemented, order %s", algo_strings[algo80], hashOrder[0]);
        usleep(10);
        return -1;
    }

    int warn = 0;
    do {
        int order = 0;

        // 80-byte first round
        switch (algo80) {
        case BLAKE:
            quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
            TRACE("blake80:");
            break;
	case SHABAL:
	    x16_shabal512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
	   TRACE("shabal :");
	    break;
        case GROESTL:
            groestl512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
            TRACE("grstl80:");
            break;
        case JH:
            jh512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
            TRACE("jh51280:");
            break;
        case KECCAK:
            keccak512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
            TRACE("kecck80:");
            break;
        case SKEIN:
            skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], 1); order++;
            TRACE("skein80:");
            break;
	case FUGUE:
	    x16_fugue512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
	    TRACE("fugue  :");
	    break;
        case CUBEHASH:
            cubehash512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
            TRACE("cube 80:");
            break;
        case SIMD:
            x16_simd512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
            TRACE("simd512:");
            break;
        case ECHO:
            x16_echo512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
            TRACE("echo   :");
            break;
        case HAMSI:
            x16_hamsi512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
            TRACE("hamsi  :");
            break;
	case SHA512:
	    x16_sha512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
	    TRACE("sha512 :");
	    break;
        } // switch algo80

        // Remaining 11 rounds in parallel
#pragma omp parallel
#pragma omp for nowait
        for (int i = 1; i < HASH_FUNC_COUNT; i++) {
            const char elem = hashOrder[i];
            const uint8_t algo64 = (elem >= 'A') ? (elem - 'A' + 10) : (elem - '0');

            switch (algo64) {
            case BLAKE:
                quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                TRACE("blake  :");
                break;
	   case SHABAL:
		x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("shabal :");
		break;
            case GROESTL:
                quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                TRACE("groestl:");
                break;
            case JH:
                quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                TRACE("jh512  :");
                break;
            case KECCAK:
                quark_keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]); order++;
                TRACE("keccak :");
                break;
            case SKEIN:
                quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                TRACE("skein  :");
                break;
	   case FUGUE:
		x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("fugue  :");
		break;
            case CUBEHASH:
                x11_cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
                TRACE("cube   :");
                break;
	   case SHA512:
		x17_sha512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
		TRACE("sha512 :");
		break;
            case SIMD:
                x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                TRACE("simd   :");
                break;
            case ECHO:
                if (use_compat_kernels[thr_id]) {
                    x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                } else {
                    x16_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
                }
                TRACE("echo   :");
                break;
            case HAMSI:
                x13_hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                TRACE("hamsi  :");
                break;
            } // switch algo64
        } // for i

        *hashes_done = pdata[19] - first_nonce + throughput;
        work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);

#ifdef _DEBUG
        uint32_t _ALIGN(64) dhash[8];
        be32enc(&endiandata[19], pdata[19]);
        x12r_hasht(dhash, endiandata); // implement x12-specific
        applog_hash(dhash);
        return -1;
#endif

        if (work->nonces[0] != UINT32_MAX) {
            const uint32_t Htarg = ptarget[7];
            uint32_t _ALIGN(64) vhash[8];
            be32enc(&endiandata[19], work->nonces[0]);
            x12rt_hash(vhash, endiandata); // implement x12rt_hash variant if name differs

            if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
                work->valid_nonces = 1;
                work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
                work_set_target_ratio(work, vhash);
                if (work->nonces[1] != 0) {
                    be32enc(&endiandata[19], work->nonces[1]);
                    x12rt_hash(vhash, endiandata);
                    bn_set_target_ratio(work, vhash, 1);
                    work->valid_nonces++;
                    pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
                } else {
                    pdata[19] = work->nonces[0] + 1;
                }
                return work->valid_nonces;
            } else if (vhash[7] > Htarg) {
                gpu_increment_reject(thr_id);
                algo80_fails[algo80]++;
                if (!warn) {
                    warn++;
                    pdata[19] = work->nonces[0] + 1;
                    continue;
                } else {
                    if (!opt_quiet)
                        gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU! %s %s",
                              work->nonces[0], algo_strings[algo80], hashOrder);
                    warn = 0;
                }
            }
        }

        if ((uint64_t)throughput + pdata[19] >= max_nonce) {
            pdata[19] = max_nonce;
            break;
        }

        pdata[19] += throughput;

    } while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

    *hashes_done = pdata[19] - first_nonce;
    return 0;
}

// --- cleanup ----------------------------------------------------------------
extern "C" void free_x12rt(int thr_id)
{
    if (thr_id < 0 || thr_id >= MAX_GPUS) return;
    if (!init[thr_id]) return;

    cudaDeviceSynchronize();
    cudaFree(d_hash[thr_id]);

    quark_blake512_cpu_free(thr_id);
    quark_groestl512_cpu_free(thr_id);
    x11_simd512_cpu_free(thr_id);
    x13_fugue512_cpu_free(thr_id);
    x16_fugue512_cpu_free(thr_id); // if present for shared code paths
    x13_hamsi512_cpu_free(thr_id);

    cuda_check_cpu_free(thr_id);

    cudaDeviceSynchronize();
    init[thr_id] = false;
}
