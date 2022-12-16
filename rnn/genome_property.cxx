#include "rnn/genome_property.hxx"
#include "common/arguments.hxx"

GenomeProperty::GenomeProperty() {
    bp_iterations = 10;
    dropout_probability = 0.0;
    min_recurrent_depth = 1;
    max_recurrent_depth = 10;
}

void GenomeProperty::generate_genome_property_from_arguments(const vector<string> &arguments) {

    get_argument(arguments, "--bp_iterations", true, bp_iterations);
    use_dropout = get_argument(arguments, "--dropout_probability", false, dropout_probability);

    get_argument(arguments, "--min_recurrent_depth", false, min_recurrent_depth);
    get_argument(arguments, "--max_recurrent_depth", false, max_recurrent_depth);
}

void GenomeProperty::set_genome_properties(RNN_Genome *genome) {
    genome->set_bp_iterations(bp_iterations);
    if (use_dropout) genome->enable_dropout(dropout_probability);
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