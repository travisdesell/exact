#ifndef EXACT_RANDOM_HXX
#define EXACT_RANDOM_HXX

#include <iostream>
using std::istream;
using std::ostream;

#include <random>
using std::minstd_rand0;

#include <vector>
using std::vector;

#include <type_traits>
using std::is_swappable;

// This must be defined in the header because c++
template <std::swappable T, std::uniform_random_bit_generator R>
void fisher_yates_shuffle(R &generator, vector<T> &v) {
  std::uniform_real_distribution<float> range{0.0, 1.0};

  for (int32_t i = v.size() - 1; i > 0; i--) {
    int32_t t = range(generator) * (i + 1);
    std::swap(v[t], v[i]);
  }
}

// Same here
template <std::uniform_random_bit_generator R>
float random_0_1(R &generator) {
  return ((float) generator() - (float) generator.min()) / ((float) generator.max() - (float) generator.min());
}

class NormalDistribution {
 private:
  bool generate;
  float z0;
  float z1;

 public:
  NormalDistribution();

  float random(minstd_rand0 &generator, float mu, float sigma);

  friend ostream &operator<<(ostream &os, const NormalDistribution &normal_distribution);
  friend istream &operator>>(istream &is, NormalDistribution &normal_distribution);

  bool operator==(const NormalDistribution &other) const;
  bool operator!=(const NormalDistribution &other) const;
};

#endif
