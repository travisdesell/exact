#include <iostream>

#include <cmath>

int main(int argc, char **argv) {
    double value = 1.3;
    double modifier = 1.19;

    for (int i = 0; i < 50; i++) {
        std::cout << std::hexfloat << std::exp(value) << std::endl;
        value *= modifier;
    }

    value = 0.13;
    modifier = 0.79;
    for (int i = 0; i < 50; i++) {
        std::cout << std::hexfloat << std::exp(value) << std::endl;
        value *= modifier;
    }
}
