#include "simple_stock_strategy.hxx"

#include "common/log.hxx"
#include "common/arguments.hxx"

SimpleStockStrategy::SimpleStockStrategy(const vector<string> &arguments) {
    buy_threshold = 0;
    get_argument(arguments, "--buy_threshold", false, buy_threshold);

    sell_threshold = 0;
    get_argument(arguments, "--sell_threshold", false, sell_threshold);

    money_pool = 100.0;
    get_argument(arguments, "--money_pool", false, money_pool);

    get_argument_vector(arguments, "--stocks", true, stocks);

    for (string stock : stocks) {
        purchased_shares[stock] = 0;
        bought_price[stock] = 0;
    }
}


void SimpleStockStrategy::make_move(const map<string, double> &context, const map<string, double> &forecast) {
    double current_money = money_pool;
    for (string stock : stocks) {
        current_money += purchased_shares[stock] * context.at(stock + "_PRC");
    }
    Log::info("current money: %lf\n", current_money);

    vector<string> stocks_to_buy;
    vector<string> stocks_to_sell;

    //ensure that <STOCK>_PRC is in the context and <STOCK>_RET is in the forecast
    bool missing_stock = false;
    for (string stock : stocks) {
        if (!context.contains(stock + "_PRC")) {
            Log::fatal("ERROR, user specified stock '%s' was being traded but '%s_PRC' was not found in context parameters.\n", stock.c_str(), stock.c_str());
            missing_stock = true;
        }

        if (!forecast.contains(stock + "_RET")) {
            Log::fatal("ERROR, user specified stock '%s' was being traded but '%s_RET' was not found in forecast parameters.\n", stock.c_str(), stock.c_str());
            missing_stock = true;
        }
    }
    if (missing_stock) exit(1);

    //determine which stocks we will sell, and which stocks we will buy
    for (string stock : stocks) {
        double forecasted_return = forecast.at(stock + "_RET");

        Log::info("forecasted return: %lf, buy_threshold: %lf, sell_threshold: %lf\n", forecasted_return, buy_threshold, sell_threshold);

        if (money_pool > 0 && forecasted_return > buy_threshold) {
            //buy stocks if we have money available and the forecasted return is greater than our
            //buy threshold.
            stocks_to_buy.push_back(stock);
            Log::info("\tbuying %s because money_pool (%lf) > 0 and forecasted_return (%lf) > buy_threshold (%lf)\n", stock.c_str(), money_pool, forecasted_return, buy_threshold);

        } else if (purchased_shares[stock] > 0 && forecasted_return < sell_threshold && context.at(stock + "_PRC") > bought_price[stock]) {
            //sell stock if we have shares, the forecasted return is less then our sell threshold 
            //and the sell price is greater than our buy price.
            stocks_to_sell.push_back(stock);
            Log::info("\tselling %s\n", stock.c_str());
        } else {
            Log::info("\tholding %s because purchased shares (%lf) == 0 or forecasted_return (%lf) >= sell_threshold (%lf) or price (%lf) <= bought price (%lf)\n", stock.c_str(), purchased_shares[stock], forecasted_return, sell_threshold, context.at(stock + "_PRC"), bought_price[stock]);
        }
    }

    Log::info("selling %d stocks.\n", stocks_to_sell.size());
    //first sell of stocks to sell
    for (string stock : stocks_to_sell) {
        if (purchased_shares[stock] > 0) {
            double stock_price = context.at(stock + "_PRC");
            double gain = purchased_shares[stock] * stock_price;
            money_pool += gain;

            Log::info("\tsold %lf shares of %s for %lf$\n", purchased_shares[stock], stock.c_str(), gain);

            purchased_shares[stock] = 0;
            bought_price[stock] = 0;
        }
    }

    Log::info("buying %d stocks.\n", stocks_to_buy.size());
    //if we have any money, use the money pool to buy stocks giving each an
    //equal amount of money
    if (money_pool > 0 && stocks_to_buy.size() > 0) {
        double money_per_stock = money_pool / stocks_to_buy.size();

        for (string stock : stocks_to_buy) {
            double stock_price = context.at(stock + "_PRC");
            double shares = money_per_stock / stock_price;
            purchased_shares[stock] = shares;
            bought_price[stock] = stock_price;

            Log::info("\tbought %lf shares of %s for %lf\n", shares, stock.c_str(), money_per_stock);
        }

        //we've spent all our money
        money_pool = 0;
    }
}


/*
Stock* get_state() {
}
*/
