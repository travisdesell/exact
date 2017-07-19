#include <iostream>
using std::cerr;
using std::endl;

#include <map>
using std::map;

#include <random>
using std::minstd_rand0;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "comparison.hxx"
#include "common/random.hxx"

bool are_different(string variable_name, const int &v1, const int &v2) {
    if (v1 != v2) {
        cerr << "IDENTICAL ERROR: " << variable_name << " different" << endl;
        cerr << "self: '" << v1 << "' vs. other: '" << v2 << "'" << endl;
        return true;
    }
    return false;
}

bool are_different(string variable_name, const float &v1, const float &v2) {
    if (v1 != v2) {
        cerr << "IDENTICAL ERROR: " << variable_name << " different" << endl;
        cerr << "self: '" << v1 << "' vs. other: '" << v2 << "'" << endl;
        return true;
    }
    return false;
}

bool are_different(string variable_name, const NormalDistribution &v1, const NormalDistribution &v2) {
    if (v1 != v2) {
        cerr << "IDENTICAL ERROR: " << variable_name << " different" << endl;
        cerr << "self: '" << v1 << "' vs. other: '" << v2 << "'" << endl;
        return true;
    }
    return false;
}

bool are_different(string variable_name, const minstd_rand0 &v1, const minstd_rand0 &v2) {
    if (v1 != v2) {
        cerr << "IDENTICAL ERROR: " << variable_name << " different" << endl;
        cerr << "self: '" << v1 << "' vs. other: '" << v2 << "'" << endl;
        return true;
    }
    return false;
}

bool are_different(string variable_name, const uniform_int_distribution<long> &v1, const uniform_int_distribution<long> &v2) {
    if (v1 != v2) {
        cerr << "IDENTICAL ERROR: " << variable_name << " different" << endl;
        cerr << "self: '" << v1 << "' vs. other: '" << v2 << "'" << endl;
        return true;
    }
    return false;
}

bool are_different(string variable_name, const uniform_real_distribution<float> &v1, const uniform_real_distribution<float> &v2) {
    if (v1 != v2) {
        cerr << "IDENTICAL ERROR: " << variable_name << " different" << endl;
        cerr << "self: '" << v1 << "' vs. other: '" << v2 << "'" << endl;
        return true;
    }
    return false;
}

bool are_different(string variable_name, const string &v1, const string &v2) {
    if (v1.compare(v2) != 0) {
        cerr << "IDENTICAL ERROR: " << variable_name << " different" << endl;
        cerr << "self: '" << v1 << "' vs. other: '" << v2 << "'" << endl;
        return true;
    }
    return false;
}

bool are_different(string variable_name, const vector<int> &v1, const vector<int> &v2) {
    if (v1.size() != v2.size()) {
        cerr << "IDENTICAL ERROR: vector " << variable_name << " different" << endl;
        cerr << "self." << variable_name << ".size(): " << v1.size() << " != other." << variable_name << ".size(): " << v2.size() << endl;
        return true;
    }

    for (uint32_t i = 0; i < v1.size(); i++) {
        if (v1[i] != v2[i]) {
            cerr << "IDENTICAL ERROR: vector " << variable_name << " different" << endl;
            cerr << "self." << variable_name << "[" << i << "]: " << v1[i] << " != other." << variable_name << "[" << i << "]: " << v2[i] << endl;
            return true;
        }
    }

    return false;
}


bool are_different(string variable_name, const vector<long> &v1, const vector<long> &v2) {
    if (v1.size() != v2.size()) {
        cerr << "IDENTICAL ERROR: vector " << variable_name << " different" << endl;
        cerr << "self." << variable_name << ".size(): " << v1.size() << " != other." << variable_name << ".size(): " << v2.size() << endl;
        return true;
    }

    for (uint32_t i = 0; i < v1.size(); i++) {
        if (v1[i] != v2[i]) {
            cerr << "IDENTICAL ERROR: vector " << variable_name << " different" << endl;
            cerr << "self." << variable_name << "[" << i << "]: " << v1[i] << " != other." << variable_name << "[" << i << "]: " << v2[i] << endl;
            return true;
        }
    }

    return false;
}

bool are_different(string variable_name, const map<string, int> &v1, const map<string, int> &v2) {
    if (v1.size() != v2.size()) return true;

    auto i1 = v1.begin();
    auto i2 = v2.begin();
    int32_t i = 0;
    while (i1 != v1.end() && i2 != v2.end()) {
        if (i1->first.compare(i2->first) != 0) {
            cerr << "IDENTICAL ERROR: map " << variable_name << " array key " << i << " is different" << endl;
            cerr << "self key '" << i1->first << "' vs other key '" << i2->first << "'" << endl;
            return true;
        }

        if (i1->second != i2->second) {
            cerr << "IDENTICAL ERROR: map " << variable_name << " array value " << i << " is different, key is: '" << i1->first << "'" << endl;
            cerr << "self value '" << i1->second << "' vs other value '" << i2->second << "'" << endl;
            return true;
        }

        i++;
    }
    return false;
}

bool are_different(string variable_name, int size, const float* v1, const float* v2) {
    for (uint32_t i = 0; i < size; i++) {
        if (v1[i] != v2[i]) {
            cerr << "IDENTICAL ERROR: array " << variable_name << " different" << endl;
            cerr << "self." << variable_name << "[" << i << "]: " << v1[i] << " != other." << variable_name << "[" << i << "]: " << v2[i] << endl;
            return true;
        }
    }
    return false;
}
