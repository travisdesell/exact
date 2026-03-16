#ifndef STRATEGY_PORTFOLIO_ORACLE_HXX
#define STRATEGY_PORTFOLIO_ORACLE_HXX

#include <map>
using std::map;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "oracle.hxx"
#include "state.hxx"


class PortfolioOracle : public Oracle {
    private:
        double initial_money_pool;
        double previous_total_money;

        vector<string> stocks;

    public:
        /**
         * Initialize a PortfolioOracle from user provided arguments.
         *
         * \param arguments are the command line arguments.
         */
        PortfolioOracle(const vector<string> &arguments);


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
        double calculate_reward(State *state, const map<string, double> &context);
};

#endif
