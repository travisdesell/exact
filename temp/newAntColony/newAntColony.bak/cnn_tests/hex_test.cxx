#include <cstdio>

#include<iostream>
using std::cout;
using std::endl;
using std::hexfloat;
using std::defaultfloat;

#include <iomanip>
using std::setprecision;

#include <sstream>
using std::ostringstream;
using std::istringstream;

#include <string>
using std::string;

int main(int argc, char** argv) {
    
    double value = 2342.092834927394e-100;

    cout << setprecision(20) << value << endl;
    cout << hexfloat << value << " stuff " << value << endl;

    printf("hex: %a\n", value);
    printf("hex: %A\n", value);

    double in_value = 0;

    char test[256];

    sprintf(test, "%a\n", value);
    sscanf(test, "%la", &in_value);

    cout << "in value: " << defaultfloat << setprecision(20) << in_value << endl;

    ostringstream oss;
    oss << hexfloat << value;

    istringstream iss(oss.str());

    double in_value_2 = 5;
    iss >> in_value_2;
    cout << "in value 2: " << setprecision(20) << in_value_2 << endl;

    double val_str = stod(oss.str());
    cout << "val_str: " << val_str << endl;
}
