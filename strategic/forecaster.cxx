#include "common/arguments.hxx"
#include "common/log.hxx"

#include "forecaster.hxx"
#include "trivial_forecaster.hxx"

Forecaster::Forecaster(const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names) : input_parameter_names(_input_parameter_names), output_parameter_names(_output_parameter_names) {
}

vector<string> Forecaster::get_output_parameter_names() const {
    return output_parameter_names;
}

vector<string> Forecaster::get_input_parameter_names() const {
    return input_parameter_names;
}

Forecaster* Forecaster::initialize_from_arguments(const vector<string> &arguments) {

    string forecaster_type;
    get_argument(arguments, "--forecaster_type", true, forecaster_type);

    if (forecaster_type == "trivial") {
        vector<string> input_parameter_names, output_parameter_names;
        get_argument_vector(arguments, "--input_parameter_names", true, input_parameter_names);
        get_argument_vector(arguments, "--output_parameter_names", true, output_parameter_names);

        return new TrivialForecaster(arguments, input_parameter_names, output_parameter_names);

    } else {
        Log::fatal("unknown forecaster type '%s', cannot evaluate strategy.\n\n", forecaster_type.c_str());
        exit(1);
    }

}
