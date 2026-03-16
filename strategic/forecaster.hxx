#ifndef FORECASTER_HXX
#define FORECASTER_HXX

#include <map>
using std::map;

#include <string>
using std::string;

#include <vector>
using std::vector;



class Forecaster {
    protected:
        vector<string> input_parameter_names;
        vector<string> output_parameter_names;

    public:
        /**
         * Initializes the abstract class of forecaster by setting its input and output parameter names
         * \param input_parameter_names are the input columns to use by the forecaster
         * \param output_parameter names are the columns forecasted by the forecaster
         */
        Forecaster(const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names);

        /**
         * Provide a virtual destructor for this virtual class
         */
        virtual ~Forecaster() = default;

        /**
         * \return the input parameter names for the forecaster
         */
        vector<string> get_input_parameter_names() const;

        /**
         * \return the output parameter names for the forecaster
         */
        vector<string> get_output_parameter_names() const;

        /**
         * Creates one of any of the potential sublcasses of the Forecaster class given
         * command line arguments.
         *
         * \param arguments is the vector of command line arguments
         *
         * \return a pointer to a Forecaster object
         */
        static Forecaster* initialize_from_arguments(const vector<string> &arguments);

        /**
         * Given the current context of the system, provide a forecast of what the next
         * context is expected to be.
         *
         * \param context is a representation of the current context of the problem being worked 
         *      on, e.g., the current time series values (used to predict the next or other future
         *      time series values).
         *
         * \return the forecasted vector of values
         */
        virtual map<string, double> forecast(const map<string, double> &context) = 0;

};

#endif
