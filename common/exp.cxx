#include <cstdint>

#include "exp.hxx"

#define iterations 50

//calculate exp slowly using a taylor series to prevent
//os/compiler inconsistencies

double exact_exp(double z) {
    bool is_negative = z < 0;
    if (is_negative) z = -z;

    // exp(x) = sum (k = 0 to inf) z^k/k!
    double result = 1.0 + z;

    double prev = z;
    for (uint32_t k = 2; k < iterations; k++) {
        prev *= (z / k);
        result += prev;
    }

    if (is_negative) {
        return 1.0 / result;
    } else {
        return result;
    }
}


