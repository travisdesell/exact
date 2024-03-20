#include "annealing.hxx"

#include <cmath>
#include <memory>

#include "common/arguments.hxx"
#include "common/log.hxx"

unique_ptr<AnnealingPolicy> AnnealingPolicy::from_arguments(const vector<string>& arguments) {
    string type;
    get_argument(arguments, "--annealing_policy", false, type);
    Log::info("Annealing policy = %s\n", type.c_str());
    if (type == "linear") {
        return unique_ptr<AnnealingPolicy>(new LinearAnnealingPolicy(arguments));
    } else if (type == "inv_exp") {
        return unique_ptr<AnnealingPolicy>(new InvExpAnnealingPolicy(arguments));
    } else if (type == "sin") {
        return unique_ptr<AnnealingPolicy>(new SinAnnealingPolicy(arguments));
    } else {
        Log::info("Using default annealing policy\n");
        return make_unique<AnnealingPolicy>();
    }
}

double AnnealingPolicy::operator()(int32_t genome_number) {
    return 0.0;
}

LinearAnnealingPolicy::LinearAnnealingPolicy(
    double start_value, double end_value, int32_t start_genomes, int32_t interp_genomes
)
    : start_value(start_value), end_value(end_value), start_genomes(start_genomes), interp_genomes(interp_genomes) {
}

LinearAnnealingPolicy::LinearAnnealingPolicy(const vector<string>& arguments) {
    get_argument(arguments, "--linear_start_value", true, start_value);
    get_argument(arguments, "--linear_end_value", true, end_value);
    get_argument(arguments, "--linear_start_genomes", true, start_genomes);
    get_argument(arguments, "--linear_interp_genomes", true, interp_genomes);
}

double LinearAnnealingPolicy::operator()(int32_t genome_number) {
    if (genome_number <= start_genomes) {
        return start_value;
    } else if (genome_number <= interp_genomes + start_genomes) {
        double weight = (double) (genome_number - (interp_genomes + start_genomes)) / (double) interp_genomes;
        return weight * end_value + (1 - weight) * start_value;
    } else {
        return end_value;
    }
}

InvExpAnnealingPolicy::InvExpAnnealingPolicy(double decay_factor) : decay_factor(decay_factor) {
}
InvExpAnnealingPolicy::InvExpAnnealingPolicy(const vector<string>& arguments) {
    get_argument(arguments, "--exp_decay_factor", true, decay_factor);
}

double InvExpAnnealingPolicy::operator()(int32_t genome_number) {
    return std::pow(1. + genome_number, -decay_factor);
}

SinAnnealingPolicy::SinAnnealingPolicy(double period, double min_p, double max_p)
    : period(period), min_p(min_p), max_p(max_p) {
    if (min_p > max_p) {
        std::swap(min_p, max_p);
    }

    if (min_p > 1.0 || min_p < 0.0) {
        throw "Invalid min_p supplied to SinAnnealingPolicyConstructor";
    }
    if (max_p > 1.0 || max_p < 0.0) {
        throw "Invalid max_p supplied to SinAnnealingPolicyConstructor";
    }
}
SinAnnealingPolicy::SinAnnealingPolicy(const vector<string>& arguments) {
    get_argument(arguments, "--sin_min_p", true, min_p);
    get_argument(arguments, "--sin_max_p", true, max_p);
    get_argument(arguments, "--sin_period", true, period);
}

double SinAnnealingPolicy::operator()(int32_t genome_number) {
    double range = max_p - min_p;

    return (max_p + min_p) / 2. + range / 2. * std::sin(2. * M_PI * genome_number / period);
}
