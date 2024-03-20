#ifndef BANDIT_STRATEGY_HXX
#define BANDIT_STRATEGY_HXX

#include <iostream>
#include <vector>
#include <map>
#include "strategy.hxx"
#include "bandit.hxx"

/**
 * Bandit Stratefy to handle stock buy/sell
 * for single company
 * TODO: update for handlin multiple companies
*/
class BanditStrategy: public Strategy {

    private:
        int n_arms;
        int dimension;
        double alpha = 1.0;
        string type;
        Bandit *bandit;
        int last_choice;

        //flag to check buy or sell cycle
        bool buy_flag = true;
        //bool holdoff_flag = false;

        double buy_threshold;

        vector<int> buy_step{0, 2};
        vector<int> sell_step{1, 2};

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
        BanditStrategy(const vector<string> &arguments) ;
        void make_move(const map<string, double> &context, const map<string, double> &forecast);

        State* get_state();

        void report_reward(double reward);

};

#endif