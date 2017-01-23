#include <iostream>
#include <iomanip>

#include <cmath>

#define iterations 101

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
    double value;
    double modifier;

    /*
    std::cout << "TESTING LARGE VALUES!" << std::endl;
    value = 1.3;
    modifier = 1.19;
    for (int i = 0; i < 50; i++) {
        std::cout << "std   " << std::hexfloat << std::exp(value) << std::endl;
        std::cout << "exact " << std::hexfloat << exact_exp(value) << std::endl;
        std::cout << "std   " << std::defaultfloat << std::setprecision(20) << std::exp(value) << std::endl;
        std::cout << "exact " << std::defaultfloat << std::setprecision(20) << exact_exp(value) << std::endl;
        std::cout << std::endl;
        value *= modifier;
    }
    std::cout << std::endl;
    */

    std::cout << "TESTING NEGATIVE LARGE VALUES!" << std::endl;
    value = -1.3;
    modifier = 1.19;
    for (int i = 0; i < 50; i++) {
        std::cout << "value " << std::defaultfloat << value << std::endl;
        std::cout << "std   " << std::hexfloat << std::exp(value) << std::endl;
        std::cout << "exact " << std::hexfloat << exact_exp(value) << std::endl;
        std::cout << "std   " << std::defaultfloat << std::setprecision(20) << std::exp(value) << std::endl;
        std::cout << "exact " << std::defaultfloat << std::setprecision(20) << exact_exp(value) << std::endl;
        std::cout << std::endl;
        value *= modifier;
    }
    std::cout << std::endl;

    /*
    std::cout << "TESTING SMALL VALUES!" << std::endl;
    value = 0.13;
    modifier = 0.079;
    for (int i = 0; i < 50; i++) {
        std::cout << "std   " << std::hexfloat << std::exp(value) << std::endl;
        std::cout << "exact " << std::hexfloat << exact_exp(value) << std::endl;
        std::cout << "std   " << std::defaultfloat << std::setprecision(20) << std::exp(value) << std::endl;
        std::cout << "exact " << std::defaultfloat << std::setprecision(20) << exact_exp(value) << std::endl;
        std::cout << std::endl;
        value *= modifier;
    }
    std::cout << std::endl;

    std::cout << "TESTING SMALL NEGATIVE VALUES!" << std::endl;
    value = -0.13;
    modifier = 0.079;
    for (int i = 0; i < 50; i++) {
        std::cout << "std   " << std::hexfloat << std::exp(value) << std::endl;
        std::cout << "exact " << std::hexfloat << exact_exp(value) << std::endl;
        std::cout << "std   " << std::defaultfloat << std::setprecision(20) << std::exp(value) << std::endl;
        std::cout << "exact " << std::defaultfloat << std::setprecision(20) << exact_exp(value) << std::endl;
        std::cout << std::endl;
        value *= modifier;
    }
    std::cout << std::endl;
    */

}
