#ifndef TRIVIAL_FORECASTER_HXX
#define TRIVIAL_FORECASTER_HXX

#include <map>
using std::map;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "forecaster.hxx"

class TrivialForecaster : public Forecaster {
    private:
        int32_t forecaster_lag;

        vector<map<string, double> > history;

    public:
        /**
         * Initialize a TrivialForecaster from user provided arguments.
         *
         * \param arguments are the command line arguments.
         */
        TrivialForecaster(const vector<string> &arguments, const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names);

        /**
         * Will return a forecast, which will be the context N contexts ago,
         * where N is the forecaster lag. If forecaster lag is 0, this will
         * return the given context. If forecaster lag is 1, it will return the
         * previous, if forecaster lag is 2, it will return the 2nd previous
         * context, etc.
         *
         * \param the context (the current values of the system for the model to
         *      provide a forecast from).
         *
         * \return a forecast given the context. 
         */
        map<string, double> forecast(const map<string, double> &context);

};


#endif
