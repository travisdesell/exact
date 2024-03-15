#include "common/arguments.hxx"
#include "common/log.hxx"

#include "strategy.hxx"
#include "simple_stock_strategy.hxx"
#include "bandit_strategy.hxx"

Strategy* Strategy::initialize_from_arguments(const vector<string> &arguments) {

    string strategy_type;
    get_argument(arguments, "--strategy_type", true, strategy_type);

    if (strategy_type == "simple_stock") {
        return new SimpleStockStrategy(arguments);
    } else if (strategy_type == "bandit") {
        Log::info("Selected bandit Strategy\n");
        return new BanditStrategy(arguments);
    } else {
        Log::fatal("unknown strategy type '%s', cannot evaluate strategy.\n\n", strategy_type.c_str());
        exit(1);
    }

}
