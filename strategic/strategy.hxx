#ifndef STRATEGY_HXX
#define STRATEGY_HXX

#include <map>
using std::map;

#include <string>
using std::string;

#include <vector>
using std::vector;

//#include "state.hxx"

class Strategy {
    public:
        /**
         * Creates one of any of the potential sublcasses of the Strategy class given
         * command line arguments.
         *
         * \param arguments is the vector of command line arguments
         *
         * \return a pointer to a Strategy object
         */
        static Strategy* initialize_from_arguments(const vector<string> &arguments);

        /**
         * Take an action or actions using this strategy given the current context and a forecast
         * of the next context.
         *
         * \param context is a representation of the current context of the problem being worked 
         *      on, e.g., the current time series values (used to predict the next or other future
         *      time series values).
         *
         * \param forecast is a forecast of the next context of the problem being worked on (e.g.,
         *      the time series values at the next time step).
         */
        virtual void make_move(const map<string,double> &context, const map<string,double> &forecast) = 0;

        /**
         * Returns the current state of the strategy so an Oracle object can calculate
         * a reward given the current state and the current context of the system.
         */
        //virtual State* get_state() = 0;
};


#endif
