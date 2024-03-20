#include "common/log.hxx"
#include "common/arguments.hxx"

#include "portfolio.hxx"
#include "portfolio_oracle.hxx"
#include "state.hxx"

PortfolioOracle::PortfolioOracle(const vector<string> &arguments) {
    //use the money pool command line argument to determine how much
    //money was started with
    initial_money_pool = 100.0;
    get_argument(arguments, "--money_pool", false, initial_money_pool);
    
    //use the initial money pool to calculate the first previous money
    //pool
    previous_total_money = initial_money_pool;

}

double PortfolioOracle::calculate_reward(State *state, const map<string, double> &context) {
    Portfolio* portfolio = dynamic_cast<Portfolio*>(state);
    if (portfolio == NULL) {
        Log::fatal("ERROR: could not cast the state object to a Portfolio. An incompatible strategy was used.\n");
        exit(1);
    }

    double current_total_money = portfolio->calculate_value(context);

    Log::info("Previous money: %2.5f\n", previous_total_money);
    Log::info("Current money: %2.5f\n", current_total_money);


    double reward = current_total_money - previous_total_money;

    previous_total_money = current_total_money;

    return reward;
}
