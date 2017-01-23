#include <cstdint>

#include "exp.hxx"

#define iterations 51

//calculate exp slowly using a taylor series to prevent
//os/compiler inconsistencies

double exact_exp(double z) {
    // exp(x) = sum (k = 0 to inf) z^k/k!
    double result = 1.0 + z;

    double zk = z;
    double k_fac = 1.0;
    for (uint32_t k = 2; k < iterations; k++) {
        zk = zk * z;

        k_fac *= k;

        result += zk/k_fac;
    }   
    return result;
}
