#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <wmmintrin.h>  // PCLMULQDQ
#include <x86intrin.h>

// Parâmetros da curva secp256k1
typedef struct {
    uint64_t x[4];  // 256 bits (4x 64 bits)
    uint64_t y[4];
} Point;

// Campo primo p (2^256 - 2^32 - 977)
static const uint64_t p[4] = {
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFEFFFF
};

// Parâmetros de Montgomery
static const uint64_t R[4] = {
    0x0000000000000001,  // R = 2^256 mod p
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000
};
static const uint64_t p_dash = 0xE865C2C25C3003EA;  // -p^-1 mod 2^64 (correto)

// Ponto gerador G (em Montgomery domain)
static const Point G_montgomery = {
    .x = {
        0x79BE667EF9DCBBAC,
        0x55A06295CE870B07,
        0x029BFCDB2DCE28D9,
        0x59F2815B16F81798
    },
    .y = {
        0x483ADA7726A3C465,
        0x5DA4FBFC0E1108A8,
        0xFD17B448A6855419,
        0x9C47D08FFB10D4B8
    }
};

// Funções auxiliares
static void mod_add(uint64_t *result, const uint64_t *a, const uint64_t *b);
static void mod_sub(uint64_t *result, const uint64_t *a, const uint64_t *b);
static void mod_inv(uint64_t *result, const uint64_t *a);
static void montgomery_reduce(uint64_t *result, const uint64_t *product);
static void montgomery_mul(uint64_t *result, const uint64_t *a, const uint64_t *b);
static void point_double(Point *result, const Point *p);
static void point_add(Point *result, const Point *p, const Point *q);
static void scalar_mult(Point *result, const uint64_t *k, const Point *g);
static void hex_to_bytes(const char *hex, uint8_t *bytes);
static void print_usage(const char *prog_name);

int main(int argc, char *argv[]) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Converter chave privada hex para bytes (com padding)
    uint8_t private_key_bytes[32] = {0};
    hex_to_bytes(argv[1], private_key_bytes);
    uint64_t k[4];
    memcpy(k, private_key_bytes, 32);

    // Calcular chave pública
    Point public_key;
    scalar_mult(&public_key, k, &G_montgomery);

    // Converter para formato comprimido
    printf("Chave pública comprimida: ");
    if (public_key.y[0] % 2 == 0) {
        printf("02");
    } else {
        printf("03");
    }
    for (int i = 0; i < 4; i++) {
        printf("%016lx", public_key.x[i]);
    }
    printf("\n");

    return 0;
}

// Montgomery Reduction usando PCLMULQDQ
static void montgomery_reduce(uint64_t *result, const uint64_t *product) {
    __m128i t[8];
    memcpy(t, product, 64);  // Copia 512 bits (8x64)

    for (int i = 0; i < 4; i++) {
        uint64_t scalar_t = _mm_extract_epi64(t[i], 0);
        uint64_t k = (scalar_t * p_dash) & 0xFFFFFFFFFFFFFFFF;
        __m128i k_vec = _mm_cvtsi64_si128(k);
        __m128i p_vec = _mm_loadu_si128((__m128i*)&p[i]);
        __m128i q = _mm_clmulepi64_si128(k_vec, p_vec, 0x00);
        __m128i t_vec = _mm_load_si128(&t[i]);
        t_vec = _mm_xor_si128(t_vec, q);
        _mm_store_si128(&t[i], t_vec);
    }

    // Resultado final (256 bits)
    memcpy(result, t + 4, 32);
}

// Multiplicação de Montgomery
static void montgomery_mul(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    __uint128_t product[8] = {0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            __uint128_t temp = (__uint128_t)a[i] * b[j];
            product[i + j] += temp;
            product[i + j + 1] += product[i + j] >> 64;
            product[i + j] &= 0xFFFFFFFFFFFFFFFF;
        }
    }
    montgomery_reduce(result, (uint64_t*)product);
}


// Inversão modular usando Fermat's Little Theorem
static void mod_inv(uint64_t *result, const uint64_t *a) {
    uint64_t exponent[4] = {
        0xFFFFFFFFFFFEFFFF,  // p - 2 (parte menos significativa)
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF
    };
    memcpy(result, a, 32);
    for (int i = 255; i >= 0; i--) {
        montgomery_mul(result, result, result);
        if ((exponent[i / 64] >> (i % 64)) & 1) {
            montgomery_mul(result, result, a);
        }
    }
}

// Adição modular
static void mod_add(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    __m128i carry = _mm_setzero_si128();
    for (int i = 0; i < 4; i += 2) {
        __m128i ai = _mm_loadu_si128((__m128i*)&a[i]);
        __m128i bi = _mm_loadu_si128((__m128i*)&b[i]);
        __m128i sum = _mm_add_epi64(ai, bi);
        sum = _mm_add_epi64(sum, carry);
        _mm_storeu_si128((__m128i*)&result[i], sum);
        carry = _mm_srli_epi64(sum, 63);
    }
    for (int i = 0; i < 4; i++) {
        if (result[i] >= p[i]) {
            result[i] -= p[i];
        }
    }
}

// Subtração modular
static void mod_sub(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    __m128i borrow = _mm_setzero_si128();
    for (int i = 0; i < 4; i += 2) {
        __m128i ai = _mm_loadu_si128((__m128i*)&a[i]);
        __m128i bi = _mm_loadu_si128((__m128i*)&b[i]);
        __m128i diff = _mm_sub_epi64(ai, bi);
        diff = _mm_sub_epi64(diff, borrow);
        _mm_storeu_si128((__m128i*)&result[i], diff);
        borrow = _mm_srli_epi64(diff, 63);
    }
    for (int i = 0; i < 4; i++) {
        if (result[i] > p[i]) {
            result[i] += p[i];
        }
    }
}

// Dobramento de ponto
static void point_double(Point *result, const Point *p) {
    uint64_t x_squared[4], three_x_squared[4], two_y[4], lambda[4];
    montgomery_mul(x_squared, p->x, p->x);
    montgomery_mul(three_x_squared, x_squared, (uint64_t[]){3, 0, 0, 0});
    montgomery_mul(two_y, p->y, (uint64_t[]){2, 0, 0, 0});
    mod_inv(lambda, two_y);  // Correção: usar inversão modular
    montgomery_mul(lambda, three_x_squared, lambda);

    uint64_t lambda_sq[4], two_x[4];
    montgomery_mul(lambda_sq, lambda, lambda);
    montgomery_mul(two_x, p->x, (uint64_t[]){2, 0, 0, 0});
    mod_sub(result->x, lambda_sq, two_x);

    uint64_t x_diff[4], lambda_x_diff[4];
    mod_sub(x_diff, p->x, result->x);
    montgomery_mul(lambda_x_diff, lambda, x_diff);
    mod_sub(result->y, lambda_x_diff, p->y);
}

// Adição de pontos
static void point_add(Point *result, const Point *p, const Point *q) {
    if (memcmp(p, q, sizeof(Point)) == 0) {
        point_double(result, p);
        return;
    }

    uint64_t y_diff[4], x_diff[4], lambda[4];
    mod_sub(y_diff, q->y, p->y);
    mod_sub(x_diff, q->x, p->x);
    mod_inv(lambda, x_diff);  // Correção: usar inversão modular
    montgomery_mul(lambda, y_diff, lambda);

    uint64_t lambda_sq[4], x_sum[4];
    montgomery_mul(lambda_sq, lambda, lambda);
    mod_add(x_sum, p->x, q->x);
    mod_sub(result->x, lambda_sq, x_sum);

    uint64_t x_p_diff[4], lambda_x_p_diff[4];
    mod_sub(x_p_diff, p->x, result->x);
    montgomery_mul(lambda_x_p_diff, lambda, x_p_diff);
    mod_sub(result->y, lambda_x_p_diff, p->y);
}

// Multiplicação escalar
static void scalar_mult(Point *result, const uint64_t *k, const Point *g) {
    Point addend = *g;
    Point temp = {0};
    for (int i = 255; i >= 0; i--) {
        point_double(&temp, &temp);
        if ((k[i / 64] >> (i % 64)) & 1) {
            point_add(&temp, &temp, &addend);
        }
    }
    *result = temp;
}

// Converter hex string para bytes (com padding)
static void hex_to_bytes(const char *hex, uint8_t *bytes) {
    int len = strlen(hex);
    memset(bytes, 0, 32);  // Preenche com zeros

    // Copia os bytes da direita para a esquerda
    for (int i = 0; i < len; i += 2) {
        sscanf(hex + len - i - 2, "%2hhx", &bytes[31 - i/2]);
    }
}

// Mensagem de uso
static void print_usage(const char *prog_name) {
    printf("Uso: %s <chave_privada_hex>\n", prog_name);
    printf("Exemplo: %s 3\n", prog_name);
}

// gcc -O3 -msse4.2 -mpclmul -o mult_mod mult_mod.c
