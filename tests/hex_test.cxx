#include<iostream>
using std::cout;
using std::endl;

#include <iomanip>
using std::setprecision;

int main(int argc, char** argv) {
    
    double value = 2342.092834927394;

    cout << std::hex << value << endl;
    cout << value << endl;
    cout << std::dec << value << endl;
    cout << value << endl;

    printf("hex: %a\n", value);
    printf("hex: %A\n", value);

    double in_value = 0;

    char test[256];

    sprintf(test, "%a\n", value);
    sscanf(test, "%la", &in_value);

    cout << "in value: " << setprecision(20) << in_value << endl;
}
