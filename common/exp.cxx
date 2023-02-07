#ifdef EXACT_MATH_TEST
#include <cmath>
#include <cstdlib>

#include <iostream>
using std::cout;
using std::endl;
#endif

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


float exact_sqrt(float s) {
    if (s == 0) return 0.0;

    float s_prev = s;

    float s_next = 0.5 * (s_prev + (s / s_prev));
    s_prev = s_next;
    s_next = 0.5 * (s_prev + (s / s_prev));
    s_prev = s_next;
    s_next = 0.5 * (s_prev + (s / s_prev));
    s_prev = s_next;
    s_next = 0.5 * (s_prev + (s / s_prev));
    s_prev = s_next;
    s_next = 0.5 * (s_prev + (s / s_prev));
    s_prev = s_next;
    s_next = 0.5 * (s_prev + (s / s_prev));
    s_prev = s_next;
    s_next = 0.5 * (s_prev + (s / s_prev));

    return s_next;
}

#ifdef EXACT_MATH_TEST

int main(int argc, char **argv) {
    
    float value = 0.932;

    for (int i = 0; i < 100; i++) {
        float exact_result = exact_sqrt(value);
        float default_result = sqrt(value);

        cout << "value: " << value << ", exact_sqrt: " << exact_result << ", sqrt: " << default_result << endl;
        value += drand48();
    }
}

#endif
