#include <iostream>
#include <iomanip>

#include <cmath>

#define iterations 30

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


int main(int argc, char **argv) {
    double value = 1.3;
    double modifier = 1.19;

    for (int i = 0; i < 50; i++) {
        std::cout << std::hexfloat << std::exp(value) << std::endl;
        std::cout << std::hexfloat << exact_exp(value) << std::endl;
        std::cout << std::defaultfloat << std::setprecision(20) << std::exp(value) << std::endl;
        std::cout << std::defaultfloat << std::setprecision(20) << exact_exp(value) << std::endl;
        value *= modifier;
    }

    value = 0.13;
    modifier = 0.79;
    for (int i = 0; i < 50; i++) {
        std::cout << std::hexfloat << std::exp(value) << std::endl;
        std::cout << std::hexfloat << exact_exp(value) << std::endl;
        std::cout << std::defaultfloat << std::setprecision(20) << std::exp(value) << std::endl;
        std::cout << std::defaultfloat << std::setprecision(20) << exact_exp(value) << std::endl;
        value *= modifier;
    }
}
