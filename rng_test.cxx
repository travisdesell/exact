#include <iostream>
using std::cout;
using std::endl;

#include <random>
using std::mt19937;
using std::default_random_engine;
using std::minstd_rand0;


int main(int argc, char **argv) {
    long seed = 1337;

    mt19937 g1(seed);

    cout << g1 << endl;
    for (uint32_t i = 0; i < 5; i++) {
        cout << "\t" << g1() << endl;
    }
    cout << g1 << endl;
    cout << endl << endl;

    default_random_engine g2(seed);
    cout << g2 << endl;
    for (uint32_t i = 0; i < 5; i++) {
        cout << "\t" << g2() << endl;
    }
    cout << g2 << endl;
    cout << endl << endl;

    minstd_rand0 g3(seed);
    cout << g3 << endl;
    for (uint32_t i = 0; i < 5; i++) {
        cout << "\t" << g3() << endl;
    }
    cout << g3 << endl;
    cout << endl << endl;
}
