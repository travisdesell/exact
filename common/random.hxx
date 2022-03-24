#ifndef EXACT_RANDOM_HXX
#define EXACT_RANDOM_HXX

#include <iostream>
using std::istream;
using std::ostream;

#include <random>
using std::minstd_rand0;

#include <vector>
using std::vector;

// clang on macOS is weird and doesnt fully support the C++20 concepts feature, so we have to manually defined these
// concepts with GCC and non-apple clang they are defined as a part of the standard library.
#ifdef __APPLE__
#include <type_traits>

template<typename T>
concept swappable = std::is_swappable<T>::value;

template <class F, class... Args>

concept invocable = requires(F &&f, Args &&...args) {
  std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
  /* not required to be equality preserving */
};
template <class G>
concept uniform_random_bit_generator = requires(G &g) {
  std::uniform_int_distribution(0, 10)(g);
};
>>>>>>> ceb3fda518535501528c0177982183dcdc1b3983

#else
#include <concepts>
using std::swappable;
using std::uniform_random_bit_generator;
#endif

// This must be defined in the header because c++
template <swappable T, uniform_random_bit_generator R>
void fisher_yates_shuffle(R &generator, vector<T> &v) {
  std::uniform_real_distribution<float> range{0.0, 1.0};

  for (int32_t i = v.size() - 1; i > 0; i--) {
    int32_t t = range(generator) * (i + 1);
    std::swap(v[t], v[i]);
  }
}

// Same here
template <uniform_random_bit_generator R>
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
