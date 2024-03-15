#include <iostream>
#include <vector>
#include <map>
#include "common/log.hxx"
#include "common/arguments.hxx"
#include "linucb.cxx"

#include "bandit_strategy.hxx"
#include "portfolio.hxx"

string pr_suffix = "_PRC";
string rt_suffix = "_predicted_RET";

BanditStrategy::BanditStrategy(const vector<string> &arguments) {
            n_arms = 3;
            //get_argument(arguments, "--n_arms", false, n_arms);

            dimension = 7;
            get_argument(arguments, "--dimension", false, dimension);

            alpha = 1.0;
            get_argument(arguments, "--alpha", false, alpha);

            type = "linucb";
            get_argument(arguments, "--type", false, type);

            buy_threshold = 0;
            get_argument(arguments, "--buy_threshold", false, buy_threshold);

            sell_threshold = 0;
            get_argument(arguments, "--sell_threshold", false, sell_threshold);

            money_pool = 100.0;
            get_argument(arguments, "--money_pool", false, money_pool);

            get_argument_vector(arguments, "--stocks", true, stocks);

            if (type == "linucb") { 
                Log::info("Selected type LinUCB\n");
                bandit = new LinUCB(n_arms, dimension, alpha);
            } else {
                cout << "Not supported" << endl;
            }

            for (string stock : stocks) {
                purchased_shares[stock] = 0;
                bought_price[stock] = 0;
            }
        }

void BanditStrategy::make_move(const map<string, double> &context, const map<string, double> &forecast){

            double current_money = money_pool;
            for (string stock : stocks) {
                current_money += purchased_shares[stock] * context.at(stock + pr_suffix);
            }
            Log::info("current money: %lf\n", current_money);

            vector<string> stocks_to_buy;
            vector<string> stocks_to_sell;

            bool missing_stock = false;
            for (string stock : stocks) {

                if (context.find(stock + pr_suffix) == context.end()) {
                //if (!context.contains(stock + pr_suffix)) {
                    Log::fatal("ERROR, user specified stock '%s' was being traded but '%s_PRC' was not found in context parameters.\n", stock.c_str(), stock.c_str());
                    missing_stock = true;
                }

                if (context.find(stock + rt_suffix) == context.end()) {
                //if (!forecast.contains(stock + rt_suffix)) {
                    Log::fatal("ERROR, user specified stock '%s' was being traded but '%s_RET' was not found in forecast parameters.\n", stock.c_str(), stock.c_str());
                    missing_stock = true;
                }
            }

            if (missing_stock) exit(1);

            int n = context.size();
            double contextValues[n];
            int i = 0;
            for (const auto it: context) {
                contextValues[i] = it.second;
                ++i;
            }

            int choice;

            if (buy_flag) {
                choice = ((LinUCB*)bandit)->select_arm(contextValues, buy_step);
                buy_flag = false;
            } else {
                choice = ((LinUCB*)bandit)->select_arm(contextValues, sell_step);
                buy_flag = true;
            }

            //int arm = bandit->select_arm(contextValues);
            Log::info("Selected Arm is: %d ", choice);


            /**
             * If arm = 0 BUY, arm = 1 SELL, arm = 2 HOLDOFF
            */

            if (choice == 0) {
                std::cout<<"Hence BUY"<<std::endl;
            } else if (choice == 1) {
                std::cout<<"Hence SELL" <<std::endl;
            } else {
                std::cout<<"Hence HOLDOFF"<<std::endl;
            }
            this->last_choice = choice;

           for (string stock : stocks) {
            double forecasted_return = forecast.at(stock + rt_suffix);
            
            if (choice == 0 && money_pool > 0){

                stocks_to_buy.push_back(stock);

            } else if (choice == 1 && purchased_shares[stock] > 0) {

                stocks_to_sell.push_back(stock);

            }else {
                Log::debug("Actual choice: 2 HOLDOFF\n");
                this->last_choice = 2;
                Log::info("\tholding %s because purchased shares (%lf) == 0 or forecasted_return (%lf) >= sell_threshold (%lf) or price (%lf) <= bought price (%lf)\n", stock.c_str(), purchased_shares[stock], forecasted_return, sell_threshold, context.at(stock + pr_suffix), bought_price[stock]);
            }
           }

            /**
             * Update stock states based on decision
            */
           if (choice == 1) {

            Log::info("selling %d stocks.\n", stocks_to_sell.size());
            //first sell of stocks to sell
            for (string stock : stocks_to_sell) {
                if (purchased_shares[stock] > 0) {
                    double stock_price = context.at(stock + pr_suffix);
                    double gain = purchased_shares[stock] * stock_price;
                    money_pool += gain;

                    Log::info("\tsold %lf shares of %s for %lf$\n", purchased_shares[stock], stock.c_str(), gain);

                    purchased_shares[stock] = 0;
                    bought_price[stock] = 0;
                }
            }
           } else if (choice == 0) {

            Log::debug("Money pool: %5.5f\n", money_pool);
            //if we have any money, use the money pool to buy stocks giving each an
            //equal amount of money
            if (money_pool > 0 && stocks_to_buy.size() > 0) {
                Log::info("buying %d stocks.\n", stocks_to_buy.size());
                double money_per_stock = money_pool / stocks_to_buy.size();

                for (string stock : stocks_to_buy) {
                    double stock_price = context.at(stock + pr_suffix);
                    double shares = money_per_stock / stock_price;
                    purchased_shares[stock] = shares;
                    bought_price[stock] = stock_price;

                    Log::info("\tbought %lf shares of %s for %lf\n", shares, stock.c_str(), money_per_stock);
                }

                //we've spent all our money
                money_pool = 0;
            }
           }
        }

State* BanditStrategy::get_state(){
            //State *state = new BanditState(this->last_choice);
            Portfolio *portfolio = new Portfolio(money_pool);

            for (string stock : stocks) {
                portfolio->add_stock(stock, purchased_shares[stock]);
            }
            return portfolio;
        }

void BanditStrategy::report_reward(double reward) {
            //std::cout<<"Update Reward: "<<reward<<" for choice: "<<this->last_choice<<std::endl;
            this->bandit->update(reward, -1*reward, this->last_choice);
        }