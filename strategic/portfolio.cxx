#include "common/log.hxx"

#include "portfolio.hxx"


Portfolio::Portfolio(double _current_money_pool) : current_money_pool(_current_money_pool) {
}

void Portfolio::add_stock(string stock, double purchased_shares) {
    shares[stock] = purchased_shares;
}

double Portfolio::calculate_value(const map<string, double> &context) {
    double value = current_money_pool;

    for (auto const& [stock, purchased_shares] : shares) {
        Log::info("getting price for %s with %lf purchased shares.\n", stock.c_str(), purchased_shares);

        double price = context.at(stock + "_PRC");
        Log::info("price: %5.5f\n", price);

        value += price * purchased_shares;
    }

    return value;
}
