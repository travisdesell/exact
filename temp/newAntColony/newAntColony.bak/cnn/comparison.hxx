#ifndef EXACT_COMPARISON_HXX
#define EXACT_COMPARISON_HXX

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

#include "common/random.hxx"

bool are_different(string variable_name, const int &v1, const int &v2);
bool are_different(string variable_name, const float &v1, const float &v2);
bool are_different(string variable_name, const string &v1, const string &v2);
bool are_different(string variable_name, const minstd_rand0 &v1, const minstd_rand0 &v2);
bool are_different(string variable_name, const NormalDistribution &v1, const NormalDistribution &v2);
bool are_different(string variable_name, const uniform_int_distribution<long> &v1, const uniform_int_distribution<long> &v2);
bool are_different(string variable_name, const uniform_real_distribution<float> &v1, const uniform_real_distribution<float> &v2);

bool are_different(string variable_name, const vector<int> &v1, const vector<int> &v2);
bool are_different(string variable_name, const vector<long> &v1, const vector<long> &v2);

bool are_different(string variable_name, const map<string, int> &v1, const map<string, int> &v2);

bool are_different(string variable_name, int size, const float* v1, const float* v2);


/*

template <>
bool are_different<string>(string variable_name, const string &v1, const string &v2);

template <>
bool are_different<vector<long>>(string variable_name, const vector<long> &v1, const vector<long> &v2);

template <>
bool are_different<map<string, int>>(string variable_name, const map<string, int> &v1, const map<string, int> &v2);
*/

#endif
