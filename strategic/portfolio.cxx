#include "common/log.hxx"

#include "portfolio.hxx"


Portfolio::Portfolio(double _current_money_pool, string _price_suffix) : current_money_pool(_current_money_pool), price_suffix(_price_suffix) {
}

void Portfolio::add_stock(string stock, double purchased_shares) {
    shares[stock] = purchased_shares;
}

double Portfolio::calculate_value(const map<string, double> &context) {
    double value = current_money_pool;

    for (auto const& [stock, purchased_shares] : shares) {
        Log::info("getting price for %s with %lf purchased shares.\n", stock.c_str(), purchased_shares);

        double price = context.at(stock + price_suffix);

        value += price * purchased_shares;
    }

    return value;
}
