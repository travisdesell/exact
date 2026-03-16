#ifndef STRATEGY_ORACLE_HXX
#define STRATEGY_ORACLE_HXX

#include <map>
using std::map;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "state.hxx"

class Oracle {
    public:
        /**
         * Creates one of any of the potential sublcasses of the Oracle class given
         * command line arguments.
         *
         * \param arguments is the vector of command line arguments
         *
         * \return a pointer to an Oracle object
         */
        static Oracle* initialize_from_arguments(const vector<string> &arguments);


        /**
         * A default destructor for this virtual class.
         */
        virtual ~Oracle() = default;

        /**
         * Calculates the reward given the state of a strategy and the current state (context) of
         * the system.
         *
         * \param state is some subclass of the State class, representing the state of the strategy
         *      needed to calculate a reward.
         * \param context is the context of the system for the next time step after it made its
         *      last move.
         *
         * \return a reward value representing how good the strategy's current state is compared
         *      to its previous state.
         */
        virtual double calculate_reward(State *state, const map<string, double> &context) = 0;
};

#endif
