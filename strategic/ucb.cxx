#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include "bandit.hxx"
using namespace std;

class UCB: public Bandit {
    private:
     int n_arms;
     double confidence;
     string type;
     double* arm_sum_rewards;
     int N;
     int* arm_counts;
     double* ucb_values;

    public: 

        UCB(int n_arms, double confidence){
            this->n_arms = n_arms;
            this->confidence = confidence;
            this->type = "UCB";
            this->arm_counts = (int*)malloc(n_arms*sizeof(int));
            this->ucb_values = (double*)malloc(n_arms*sizeof(double));
            this->arm_sum_rewards = (double*)malloc(n_arms*sizeof(double));
            for (int i =0;i < this->n_arms; i++){
                this->arm_counts[i] = 0;
                this->ucb_values[i] = 0;
                this->arm_sum_rewards[i] = 0;
            }
        }

        int select_arm(double *context) {
            double r;
            int best_arm_index;
            int best_ucb_arm = -1;
            for(int i =0; i < this->n_arms; i++) {
                if (this->arm_counts[i] > 0) {
                    r = (this->arm_sum_rewards[i]/this->arm_counts[i]) 
                        + this->confidence * sqrt(2 * log(N)/this->arm_counts[i]);
                } else {
                    this->ucb_values[i] = INT32_MAX;
                }
            }
            for(int i = 0; i < this->n_arms; i++) {
                if (best_ucb_arm < this->ucb_values[i]) {
                    best_ucb_arm = this->ucb_values[i];
                    best_arm_index = i;
                }
            }
            return best_arm_index;
        }
        void update(double reward, double regret, int choice) {
            this->N += 1;
            this->arm_counts[choice] += 1;
            this->arm_sum_rewards[choice] += reward;
            return;
        }
};