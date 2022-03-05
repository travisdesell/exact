#include "speciation_strategy.hxx"
using std::random_device;

SpeciationStrategy::SpeciationStrategy() {
  random_device rd;
  generator = mt19937_64(rd());
  // Warm up RNG
  for (int i = 0; i < 100; i++)
    generator();
}

int32_t SpeciationStrategy::get_generated_genomes() { return generated_genomes; }

int32_t SpeciationStrategy::get_inserted_genomes() { return inserted_genomes; }
