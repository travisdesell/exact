#include <algorithm>
using std::find;

#include "trivial_forecaster.hxx"

#include "common/log.hxx"
#include "common/arguments.hxx"

TrivialForecaster::TrivialForecaster(const vector<string> &arguments, const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names) : Forecaster(_input_parameter_names, _output_parameter_names) {
    forecaster_lag = 0;
    get_argument(arguments, "--forecaster_lag", false, forecaster_lag);

    if (forecaster_lag < 0) {
        Log::fatal("ERROR: forecaster_lag must be a value >= 0, value was %d\n", forecaster_lag);
        exit(1);
    }
    
    //check to make sure all the output parameter names are available in the input parameter names
    for (string output_parameter_name : output_parameter_names) {
        if (find(input_parameter_names.begin(), input_parameter_names.end(), output_parameter_name) == input_parameter_names.end()) {
            Log::fatal("ERROR: could not find output parameter name '%s' in the input parameter names.\n", output_parameter_name.c_str());
            Log::fatal("The trivial forecaster requries all output parameter names to be in the input parameter names:\n");
            for (string input_parameter_name : input_parameter_names) {
                Log::fatal("\t%s\n", input_parameter_name.c_str());
            }
            exit(1);
        }
    }
}


map<string, double> TrivialForecaster::forecast(const map<string, double> &context) {
    if (forecaster_lag == 0) {
        // predictions for the next value(s) are just the current values
        map<string, double> result;

        //only return the output parameter names
        for (string output_parameter_name : output_parameter_names) {
            result[output_parameter_name] = context.at(output_parameter_name);
        }

        return result;
    } else {
        history.push_back(context);

        map<string, double> lagged_context = history[0];

        //only return the output parameter names
        map<string, double> result;
        for (string output_parameter_name : output_parameter_names) {
            result[output_parameter_name] = lagged_context.at(output_parameter_name);
        }

        //return the Nth previous context where N is the forecaster_lag
        if (history.size() > forecaster_lag) {
            history.erase(history.begin());
        }

        return result;
    }
}
