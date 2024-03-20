#ifndef STOCK_PORTFOLIO_HXX
#define STOCK_PORTFOLIO_HXX

#include <map>
using std::map;

#include <string>
using std::string;

#include "state.hxx"


class Portfolio : public State {
    private:
        double current_money_pool;
        map<string, double> shares;

    public:
        /**
         * Initialize the portfolio with how much money is available.
         */
        Portfolio(double _current_money_pool);

        /**
         * Used to specify how much of each stock has been purchased (if any).
         *
         * \param stock is the stock ticker name to add
         * \param purchased shares is how many shares are owned of the stock
         */
        void add_stock(string stock, double purchased_shares);

        /**
         * Calculate how much money we have available in our money pool plus the
         * values of all our stocks.
         *
         * \param context is the current state which has all the current stock ticker
         *      prices.
         *
         * \return the total worth of our stocks and current money pool
         */
        double calculate_value(const map<string, double> &context);
};

#endif
