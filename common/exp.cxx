#include <cstdint>

#include "exp.hxx"

#define iterations 50

//calculate exp slowly using a taylor series to prevent
//os/compiler inconsistencies

float exact_exp(float z) {
    bool is_negative = z < 0;
    if (is_negative) z = -z;

    // exp(x) = sum (k = 0 to inf) z^k/k!
    float result = 1.0 + z;

    float prev = z;
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


