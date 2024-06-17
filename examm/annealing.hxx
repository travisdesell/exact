#include <stdint.h>

#include <memory>
using std::unique_ptr;

#include <string>
using std::string;

#include <vector>
using std::vector;

struct AnnealingPolicy {
    static unique_ptr<AnnealingPolicy> from_arguments(const vector<string>& arguments);

    /**
     * Compute the probability to be used during genome insertion.
     * This represents the probability of inserting the genome, even if it
     * has a fitness value that is worse than the worst member in the population.
     */
    virtual double get_temperature(int32_t genome_number);

    double operator()(int32_t genome_number, double population_worst_cost, double candidate_cost);
};

/**
 * Interpolate between two values for a set number of genomes.
 * The `start_value` will be returned for `start_genomes`,
 * then a linear interpolation of `start_value` and `end_value` for
 * `interp_genomes`. Then, `end_value` is given indefinitely.
 */
class LinearAnnealingPolicy : public AnnealingPolicy {
    double start_value, end_value;
    int32_t start_genomes, interp_genomes;

   public:
    LinearAnnealingPolicy(double start_value, double end_value, int32_t start_genomes, int32_t interp_genomes);
    LinearAnnealingPolicy(const vector<string>& arguments);

    double get_temperature(int32_t genome_number) override;
};

/**
 * Calculates p by simply computing `genome_number^(-decay_factor).
 **/
class InvExpAnnealingPolicy : public AnnealingPolicy {
    double decay_factor;

   public:
    InvExpAnnealingPolicy(double decay_factor);
    InvExpAnnealingPolicy(const vector<string>& arguments);

    double get_temperature(int32_t genome_number) override;
};

/**
 * Computes `p` as a value falling on a sinusoidal curve with the supplied period.
 * a `min_p` and a `max_p` specify the range of the curve.
 **/
class SinAnnealingPolicy : public AnnealingPolicy {
    double period, min_p, max_p;

   public:
    SinAnnealingPolicy(double period, double min_p, double max_p);
    SinAnnealingPolicy(const vector<string>& arguments);

    double get_temperature(int32_t genome_number) override;
};
