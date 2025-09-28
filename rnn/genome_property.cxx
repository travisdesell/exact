#include "rnn/genome_property.hxx"

#include "common/arguments.hxx"
#include "common/log.hxx"

GenomeProperty::GenomeProperty() {
    backprop_iterations = 10;
    backprop_iterations_type = "const";
    backprop_min = 0;
    backprop_max = 10;
    backprop_exponent = 1.0;
    backprop_slope = 10;
    dropout_probability = 0.0;
    min_recurrent_depth = 1;
    max_recurrent_depth = 10;
}

void GenomeProperty::generate_genome_property_from_arguments(const vector<string>& arguments) {
    bool backprop_type = get_argument(arguments, "--backprop_iterations_type", false, backprop_iterations_type);
    if (!backprop_type) {
        backprop_iterations_type = "const";
    }
    
    if (backprop_iterations_type == "const") {
        get_argument(arguments, "--bp_iterations", true, backprop_iterations);
    }
    else if (backprop_iterations_type == "rand") {
        bool bp_min_arg = get_argument(arguments, "--bp_min", false, backprop_min);
        if (!bp_min_arg) {
            backprop_min = 0;
        }
        bool bp_max_arg = get_argument(arguments, "--bp_max", false, backprop_max);
        bool bp_iter = get_argument(arguments, "--bp_iterations", false, backprop_iterations);
        if (!bp_max_arg && !bp_iter) {
            get_argument(arguments, "--bp_max", true, backprop_max);
        }
        else if (!bp_max_arg && bp_iter) {
            backprop_max = backprop_iterations;
        }
    }
    else if (backprop_iterations_type == "scaled") {
        bool bp_min_arg = get_argument(arguments, "--bp_min", false, backprop_min);
        if (!bp_min_arg) {
            backprop_min = 0;
        }
        bool bp_max_arg = get_argument(arguments, "--bp_max", false, backprop_max);
        if (!bp_max_arg) {
            backprop_max = -1;
        }
        else if (bp_min_arg && bp_max_arg && backprop_min >= backprop_max) {
            Log::fatal("ERROR: bp_max (%d) has to be bigger than bp_min (%d)", backprop_max, backprop_min);
            exit(1);
        }
        get_argument(arguments, "--bp_exponent", true, backprop_exponent);
        get_argument(arguments, "--bp_slope", true, backprop_slope);
    }
    else {
        Log::fatal("ERROR: backprop_iterations_type is incorrectly identified. Use \"const\", \"exponentd\", \"random\", or \"acc\".");
    }
    use_dropout = get_argument(arguments, "--dropout_probability", false, dropout_probability);

    get_argument(arguments, "--min_recurrent_depth", false, min_recurrent_depth);
    get_argument(arguments, "--max_recurrent_depth", false, max_recurrent_depth);

    Log::info("Each generated genome is trained for %d epochs\n", backprop_iterations);
    Log::info("The parameters are following:\n slope: %d, exponent: %f, type: %s\n", backprop_slope, backprop_exponent, backprop_iterations_type.c_str());

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

int32_t GenomeProperty::get_backprop_max() {
    return backprop_max;
}

int32_t GenomeProperty::get_backprop_slope() {
    return backprop_slope;
}

float GenomeProperty::get_backprop_exponent() {
    return backprop_exponent;
}

string GenomeProperty::get_backprop_iterations_type() {
    return backprop_iterations_type;
}

