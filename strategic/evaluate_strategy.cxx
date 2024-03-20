#include <map>
using std::map;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/rnn_genome.hxx"
#include "time_series/time_series_new.hxx"

#include "forecaster.hxx"
#include "oracle.hxx"
#include "state.hxx"
#include "strategy.hxx"


vector<string> arguments;

void print_values(string name, const map<string, double> &values) {
    Log::debug("%s:\n", name.c_str());
    for (auto const& [key, value] : values) {
        Log::debug("\t%s = %lf\n", key.c_str(), value);
    }
}



int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", false, time_offset);

    Forecaster* forecaster = Forecaster::initialize_from_arguments(arguments);
    Strategy* strategy = Strategy::initialize_from_arguments(arguments);
    Oracle* oracle = Oracle::initialize_from_arguments(arguments);

    string time_series_filename;
    get_argument(arguments, "--time_series_filename", true, time_series_filename);
    TimeSeriesNew *time_series = new TimeSeriesNew(time_series_filename, forecaster->get_input_parameter_names(), forecaster->get_output_parameter_names());

    if (argument_exists(arguments, "--normalize")) {
        //Normalizer *normalizer = Normalizer::initialize_from_arguments(arguments);
        //normalizer->normalize(time_series_new);
    }

    double current_reward = 0.0;
    double reward = 0.0;

    map<string, double> inputs;
    map<string, double> next_inputs;

    for (int32_t i = 0; i < time_series->get_number_rows() - time_offset; i++) {
        time_series->get_inputs_at(i, inputs);
        time_series->get_inputs_at(i + time_offset, next_inputs);

        if (Log::at_level(Log::DEBUG)) print_values("inputs", inputs);

        map<string, double> forecast = forecaster->forecast(inputs);
        if (Log::at_level(Log::DEBUG)) print_values("forecast", forecast);

        strategy->make_move(inputs, forecast);

        State *state = strategy->get_state();
        current_reward = oracle->calculate_reward(state, next_inputs);
        Log::info("reward for move is: %lf\n", current_reward);

        //strategy->report_reward(current_reward);

        reward += current_reward;

        delete state;
    }

    std::cout<<"Total reward: "<<reward<<std::endl;

    delete time_series;
    delete forecaster;
    delete strategy;
    delete oracle;


    Log::release_id("main");
    return 0;
}
