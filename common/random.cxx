#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
using std::setprecision;

#include <iostream>
using std::istream;
using std::ostream;

#include <random>
using std::minstd_rand0;

#include <vector>
using std::vector;

#include "random.hxx"

NormalDistribution::NormalDistribution() {
  generate = true;
  z0 = 0;
  z1 = 0;
}

float NormalDistribution::random(minstd_rand0 &generator, float mu, float sigma) {
  const float epsilon = std::numeric_limits<float>::min();
  const float two_pi = 2.0 * 3.14159265358979323846;

  if (generate = !generate) { return z1 * sigma + mu; }

  float u1, u2;
  do {
    u1 = ((float) generator() - (float) generator.min()) / ((float) generator.max() - (float) generator.min());
    u2 = ((float) generator() - (float) generator.min()) / ((float) generator.max() - (float) generator.min());
  } while (u1 <= epsilon);

  z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
  z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
  return z0 * sigma + mu;
}

ostream &operator<<(ostream &os, const NormalDistribution &normal_distribution) {
  os << normal_distribution.generate << " " << setprecision(15) << normal_distribution.z0 << " " << setprecision(15)
     << normal_distribution.z1;

  return os;
}

istream &operator>>(istream &is, NormalDistribution &normal_distribution) {
  is >> normal_distribution.generate >> normal_distribution.z0 >> normal_distribution.z1;

  return is;
}

bool NormalDistribution::operator==(const NormalDistribution &other) const {
  return generate == other.generate && z0 == other.z0 && z1 == other.z1;
}

bool NormalDistribution::operator!=(const NormalDistribution &other) const {
  return generate != other.generate || z0 != other.z0 || z1 != other.z1;
}
