#ifndef SIMPLE_STOCK_STRATEGY_HXX
#define SIMPLE_STOCK_STRATEGY_HXX

#include <map>
using std::map;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "strategy.hxx"

class SimpleStockStrategy : public Strategy {
    private:
        //buy the stock if money is available and the predicted return
        //is above this threshold.
        double buy_threshold;

        //sell the stock of the predicted return is less than this threshold
        //but above the buying price.
        double sell_threshold;

        //how much money we have to buy stocks with initially.
        double money_pool;

        //This is used to determine which parameters need to be pulled from the
        //context and forecast. For example, if a stock name is AAPL then the
        //strategy will look for APPL_PRC (for price) from the context and 
        //APPL_RET (for returns) in the forecast.
        vector<string> stocks;

        //for each stock, track what price it was bought at the last
        //time it was bought.
        map<string, double> bought_price;

        //fore each stock, track how many shares were purchased.
        map<string, double> purchased_shares;

    public:
        /**
         * Initialize a SimpleStockStrategy from user provided arguments.
         *
         * \param arguments are the command line arguments.
         */
        SimpleStockStrategy(const vector<string> &arguments);

        /**
         * This is the simplest strategy for buying and selling stocks. In particular, a stock
         * will be purchased if the predicted return is > threshold or sold if the predicted return is
         * < threshold and the selling price is > the buying price. The default value for the threshold
         * is 0.
         *
         * \param context is a representation of the current context of the problem being worked 
         *      on, e.g., the current time series values (used to predict the next or other future
         *      time series values).
         *
         * \param forecast is a forecast of the next context of the problem being worked on (e.g.,
         *      the time series values at the next time step).
         */
        void make_move(const map<string, double> &context, const map<string, double> &forecast);

};


#endif
