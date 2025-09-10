#include "rnn/genome_property.hxx"

#include "common/arguments.hxx"
#include "common/log.hxx"

GenomeProperty::GenomeProperty() {
    backprop_iterations = 10;
    backprop_iterations_type = "const";
    backprop_min = 0;
    backprop_scale = 1.0;
    backprop_increase_genomes = 10;
    dropout_probability = 0.0;
    min_recurrent_depth = 1;
    max_recurrent_depth = 10;
}

void GenomeProperty::generate_genome_property_from_arguments(const vector<string>& arguments) {
    get_argument(arguments, "--bp_iterations", true, backprop_iterations);
    get_argument(arguments, "--backprop_iterations_type", true, backprop_iterations_type);
    bool bp_min_arg = get_argument(arguments, "--bp_min", false, backprop_min);
    bool bp_scale = get_argument(arguments, "--bp_scale", false, backprop_scale);
    get_argument(arguments, "--bp_increase_genomes", false, backprop_increase_genomes);

    use_dropout = get_argument(arguments, "--dropout_probability", false, dropout_probability);

    get_argument(arguments, "--min_recurrent_depth", false, min_recurrent_depth);
    get_argument(arguments, "--max_recurrent_depth", false, max_recurrent_depth);

    Log::info("Each generated genome is trained for %d epochs\n", backprop_iterations);
    Log::info("The parameters are following:\n increase_genomes: %d, scale: %f, type: %s\n", backprop_increase_genomes, backprop_scale, backprop_iterations_type.c_str());

    Log::info(
        "Use dropout is set to %s, dropout probability is %f\n", use_dropout ? "True" : "False", dropout_probability
    );
    Log::info("Min recurrent depth is %d, max recurrent depth is %d\n", min_recurrent_depth, max_recurrent_depth);
}

void GenomeProperty::set_genome_properties(RNN_Genome* genome) {
    Log::info("genome property backprop iterations: %d\n", backprop_iterations);

    genome->set_backprop_iterations(backprop_iterations);

    if (use_dropout) {
        genome->enable_dropout(dropout_probability);
    }
    genome->normalize_type = normalize_type;
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    genome->set_normalize_bounds(normalize_type, normalize_mins, normalize_maxs, normalize_avgs, normalize_std_devs);
}

void GenomeProperty::get_time_series_parameters(TimeSeriesSets* time_series_sets) {
    input_parameter_names = time_series_sets->get_input_parameter_names();
    output_parameter_names = time_series_sets->get_output_parameter_names();
    normalize_type = time_series_sets->get_normalize_type();
    normalize_mins = time_series_sets->get_normalize_mins();
    normalize_maxs = time_series_sets->get_normalize_maxs();
    normalize_avgs = time_series_sets->get_normalize_avgs();
    normalize_std_devs = time_series_sets->get_normalize_std_devs();
    number_inputs = time_series_sets->get_number_inputs();
    number_outputs = time_series_sets->get_number_outputs();
}

uniform_int_distribution<int32_t> GenomeProperty::get_recurrent_depth_dist() {
    return uniform_int_distribution<int32_t>(this->min_recurrent_depth, this->max_recurrent_depth);
}

void GenomeProperty::set_backprop_iterations(int32_t _backprop_iterations) {
    backprop_iterations = _backprop_iterations;
}

int32_t GenomeProperty::get_backprop_iterations() {
    return backprop_iterations;
}

int32_t GenomeProperty::get_backprop_min() {
    return backprop_min;
}

int32_t GenomeProperty::get_backprop_increase_genomes() {
    return backprop_increase_genomes;
}

float GenomeProperty::get_backprop_scale() {
    return backprop_scale;
}

string GenomeProperty::get_backprop_iterations_type() {
    return backprop_iterations_type;
}

