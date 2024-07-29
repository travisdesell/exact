#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include "bandit.hxx"
#include "common/log.hxx"
using namespace std;

/**
 * Arm of Bandit storing state variable A, b
*/
class Arm {
    private:
        int d;
        float alpha;
        double **A;
        double **A_1;
        double *b;
        double *theta;
        double *context;
        double p;

        double **res2d;
        double *res1d;


        void inv(double** mat, int order){
            double temp;
            for (int i = 0; i < order; i++) {
                for (int j = 0; j < 2 * order; j++) {
                    if (j == (i + order))
                        mat[i][j] = 1;
                }
            }
            for (int i = order - 1; i > 0; i--) {
                if (mat[i - 1][0] < mat[i][0]) {
                    double* temp = mat[i];
                    mat[i] = mat[i - 1];
                    mat[i - 1] = temp;
                }
            }
            for (int i = 0; i < order; i++) {
                for (int j = 0; j < order; j++) {
                    if (j != i) {
                        temp = mat[j][i] / mat[i][i];
                        for (int k = 0; k < 2 * order; k++) {
                        mat[j][k] -= mat[i][k] * temp;
                        }
                    }
                }
            }
            for (int i = 0; i < order; i++) {
                temp = mat[i][i];
                for (int j = 0; j < 2 * order; j++) {
                    mat[i][j] = mat[i][j] / temp;
                }
            }
        }

        double** matmul(double** a, double** b) {
            //std::cout<<"matmul 2d by 2d"<<std::endl;
            //double** c = (double**)malloc(this->d*sizeof(double*));
            //for (int i = 0; i < d; i++) {
            //    c[i] = (double*)malloc(this->d * sizeof(double));
            //    for(int j=0; j < d; j++) {
            //        c[i][j] = 0.0;
            //    }
            //}
            for(int i = 0; i < this->d; ++i){
                for(int j = 0; j < this->d; ++j){
                    res2d[i][j] = 0.0;
                    for(int k = 0; k < this->d; k++)
                        res2d[i][j] += a[i][k] * b[k][j];
                }
            }
            return res2d;
        }

        double* matmul(double** a, double* b) {
            //std::cout<<"matmul 2d by 1d"<<std::endl;
            //double* c = (double*)malloc(this->d*sizeof(double));
            for(int i = 0; i < this->d; ++i) {
                res1d[i] = 0.0;
                for(int j = 0; j < this->d; ++j){
                    res1d[i] += a[i][j] * b[i];
                }
            }
            return res1d;
        }

        double* matmul(double* a, double** b) {
            //std::cout<<"matmul 1d by 2d"<<std::endl;
            //double* c = (double*)malloc(this->d*sizeof(double));
            for(int i = 0; i < this->d; ++i) {
                res1d[i] = 0.0;
                for(int j = 0; j < this->d; ++j){
                        res1d[i] += b[i][j] * a[j];
                }
            }
            return res1d;
        }

        double** matmul(double* a, double* b) {
            //std::cout<<"matmul 1d by 1d"<<std::endl;
            //double** c = (double**)malloc(this->d*sizeof(double*));
            //for(int i = 0; i < this->d; ++i) {
            //    c[i] = (double*)malloc(this->d*sizeof(double));
            //}
            for(int i = 0; i < this->d; ++i) {
                for(int j = 0; j < this->d; ++j){
                    res2d[i][j] = a[i] * b[j];
                }
            }
            return res2d;
        }

        double dot(double *a, double* b) {
            //std::cout<<"dot 1d by 1d"<<std::endl;
            double sum = 0.0;
            for(int i = 0; i < this->d; ++i)
                sum += a[i] + b[i];
            return sum;   
        }

    public:

        Arm(int d, float alpha){
            this->d=d;
            this->alpha = alpha;

            //std::cout<<"New arm malloc"<<std::endl;
            //this->theta = (double*)malloc(d*sizeof(double));
            //std::cout<<"theta"<<std::endl;
            this->b = (double*)malloc(d*sizeof(double));
            //std::cout<<"b"<<std::endl;
            this->context = (double*)malloc(d*sizeof(double));
            //std::cout<<"context"<<std::endl;
            this->res1d = (double*)malloc(d*sizeof(double));
            //std::cout<<"res1d"<<std::endl;
            this->A = (double**)malloc(d*sizeof(double*));
            //std::cout<<"A"<<std::endl;
            this->A_1 = (double**)malloc(d*sizeof(double*));
            //std::cout<<"A_inv"<<std::endl;
            this->res2d = (double**)malloc(d*sizeof(double*));
            //std::cout<<"res2d"<<std::endl;
            for (int i = 0; i < d; i++) {
                A[i] = (double*)malloc(d * sizeof(double));
                A_1[i] = (double*)malloc(d * sizeof(double));
                res2d[i] = (double*)malloc(d * sizeof(double));
                for (int j = 0; j < d; j++) {
                    A[i][j] = 0.0;
                    res2d[i][j] = 0.0;
                }
                A[i][i] = 1.0;
                A_1[i][i] = A[i][i];
                b[i] = 0.0;
                res1d[i] = 0.0;
            }
            //std::cout<<"New arm initialized"<<std::endl;
            this->p = 0.0;
        }

        double compute_param(double* context){    
            //std::cout<<"x"<<std::endl;
            double* x = (double*)malloc(d*sizeof(double));
            for (int i = 0; i < d; i++) {
                //std::cout<<"x[i]"<<std::endl;
                x[i] = context[i];
                this->context[i] = context[i];
                //std::cout<<"Context["<<i<<"]: "<<this->context[i]<<std::endl;
                for(int j=0; j < d; j++) {
                    this->A_1[i][j] = this->A[i][j];
                }
            }
            inv(this->A_1, this->d);
            //std::cout<<"Inverse"<<std::endl;
            this->theta = matmul(A_1, b);
            //std::cout<<"Theta"<<std::endl;
            this->p = dot(this->theta, context) + alpha * sqrt(dot(matmul(x, this->A_1), x));
            //std::cout<<"p:"<<p<<std::endl;
            //for( int i = 0 ; i < d ; i++ ) {
            //    free(A_1[i]);
            //}
            //free(A_1);
            //free(x);
            return this->p;
        }

        void update_arm(double reward){
            //for (int i = 0; i < this->d; i++) {
            //    cout<<"Context["<<i<<"]: "<<this->context[i]<<endl;
            //}
            matmul(this->context, this->context);
            for(int i=0; i < this->d; i++) {
                for(int j = 0; j< this->d; j++) {
                    this->A[i][j] += res2d[i][j];
                }
                b[i] += reward*this->context[i];
            }
        }

        void printArray(double **arr) {
            for(int i = 0; i<d;i++) {
                for(int j=0;j<d;j++) {
                    std::cout<<arr[i][j]<< " ";
                }
                std::cout<<std::endl;
            }
        }

};

/**
 * Linear UCB Bandit Algorithm
*/
class LinUCB: public Bandit {
    private:
     int n_arms;
     int dim;
     double alpha;
     string type;
     double* arm_sum_rewards;
     int N;
     int* arm_counts;
     double* ucb_values;
     vector<Arm> arms;

    public: 

        LinUCB(int n_arms, int dim, double alpha){
            this->n_arms = n_arms;
            this->dim = dim;
            this->alpha = alpha;
            this->type = "LinUCB";
            this->arm_counts = (int*)malloc(n_arms*sizeof(int));
            this->ucb_values = (double*)malloc(n_arms*sizeof(double));
            this->arm_sum_rewards = (double*)malloc(n_arms*sizeof(double));
            this->arms.reserve(n_arms);
            for (int i =0;i < this->n_arms; i++){
                this->arm_counts[i] = -1;
                this->ucb_values[i] = 0;
                this->arm_sum_rewards[i] = 0;
            }
        }

        int select_arm(double* context) {
            //std::cout<<"Select Arm"<<std::endl;
            vector<int> best_arms;
            int best_arm_index;
            double best_ucb_arm = numeric_limits<double>::lowest();
            for(int i = 0; i < this->n_arms; i++) {
                //Log::info("Arm %d\n", i);
                if (this->arm_counts[i] >= 0) {
                    //std::cout<<"Compute Param"<<std::endl;
                    this->ucb_values[i] = this->arms[i].compute_param(context);
                    //std::cout<<"Computed Param"<<std::endl;
                } else {
                    //std::cout<<"New Arm"<<std::endl;
                    this->arms.push_back(Arm(dim, alpha));
                    this->arm_counts[i] += 1;
                    //std::cout<<"Pushed Back"<<std::endl;
                }
            }
            for(int i = 0; i < this->n_arms; i++) {
                //std::cout<<"\t\tArm "<<i<<": "<<this->ucb_values[i]<<std::endl;
                if (best_ucb_arm < this->ucb_values[i]) {
                    best_ucb_arm = this->ucb_values[i];
                    best_arm_index = i;
                    //cout<<"\t\t"<<best_arm_index<<endl;
                }
            }

            // Break Ties
            for(int i = 0; i < this->n_arms; i++) {
                if (best_ucb_arm == this->ucb_values[i]) {
                    best_arms.push_back(i);
                }
            }
            int random_position = rand() % best_arms.size();
            best_arm_index = best_arms[random_position];

            return best_arm_index;
        }

        int select_arm(double* context, vector<int> action_set) {
            //std::cout<<"Select Arm"<<std::endl;
            vector<int> best_arms;
            int best_arm_index;
            double best_ucb_arm = numeric_limits<double>::lowest();
            for(int i = 0; i < action_set.size(); i++) {
                //Log::info("Arm %d\n", i);
                if (this->arm_counts[action_set[i]] >= 0) {
                    //std::cout<<"Compute Param"<<std::endl;
                    this->ucb_values[action_set[i]] = this->arms[action_set[i]].compute_param(context);
                    //std::cout<<"Computed Param"<<std::endl;
                } else {
                    //std::cout<<"New Arm"<<std::endl;
                    this->arms.push_back(Arm(dim, alpha));
                    this->arm_counts[action_set[i]] += 1;
                    //std::cout<<"Pushed Back"<<std::endl;
                }
            }
            for(int i = 0; i < action_set.size(); i++) {
                //std::cout<<"\t\tArm "<<i<<": "<<this->ucb_values[i]<<std::endl;
                if (best_ucb_arm < this->ucb_values[action_set[i]]) {
                    best_ucb_arm = this->ucb_values[action_set[i]];
                    best_arm_index = action_set[i];
                    //cout<<"\t\t"<<best_arm_index<<endl;
                }
            }

            // Break Ties
            for(int i = 0; i < action_set.size(); i++) {
                if (best_ucb_arm == this->ucb_values[action_set[i]]) {
                    best_arms.push_back(action_set[i]);
                }
            }
            int random_position = rand() % best_arms.size();
            best_arm_index = best_arms[random_position];

            return best_arm_index;
        }

        void update(double reward, double regret, int choice) {
            this->N += 1;
            this->arm_counts[choice] += 1;
            //std::cout<<"Arm Count at "<<choice<<" is "<<this->arm_counts[choice]<<std::endl;
            this->arm_sum_rewards[choice] += reward;
            this->arms[choice].update_arm(reward);
            return;
        }
};