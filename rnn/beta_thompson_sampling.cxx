#include "beta_thompson_sampling.hxx"

BetaThompsonSampling::BetaThompsonSampling(int32_t _n_actions) : ThompsonSampling(_n_actions) {
    for (int32_t i = 0; i < n_actions; i++) {
        alphas.push_back(1.0);
        betas.push_back(1.0);
    }
}

BetaThompsonSampling::~BetaThompsonSampling() { }

int32_t BetaThompsonSampling::sample_action(minstd_rand0 &generator) {

    double max_theta = -1.0;
    int32_t max_theta_index = -1;

    for (int32_t i = 0; i < n_actions; i++) {
        gamma_distribution<double> alpha_dist(alphas[i], 1.0);
        gamma_distribution<double> beta_dist(betas[i], 1.0);

        double x = alpha_dist(generator);
        double y = beta_dist(generator);
        
        // Sample from beta distribution using two gamma distributions
        double theta = x / (x + y);
        if (theta > max_theta) {
            max_theta = theta;
            max_theta_index = i;
        }
    }

    return max_theta_index;

}

void BetaThompsonSampling::update(int32_t action, double reward) {
    alphas[action] += reward;
    betas[action] += 1 - reward;
#define min(x, y) x > y ? x : y
    for (int32_t i = 0; i < n_actions; i++) {
        alphas[action] = max(1.0, alphas[action] * 0.99);
        betas[action] = max(1.0, alphas[action] * 0.99);
    }
}


void BetaThompsonSampling::print(vector<int> &possible_node_types, const string node_types[]) {
    printf("BetaThompsonSampling mean rewards:\n");
    for (int32_t action = 0; action < n_actions; action++) {
        const char *node_type_str = node_types[possible_node_types[action]].c_str();
        double mean = alphas[action] / (alphas[action] + betas[action]);
        printf("    %s node expected reward = %llf\n", node_type_str, mean);
    }
}
