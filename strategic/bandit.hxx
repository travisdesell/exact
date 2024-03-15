#ifndef BANDIT_HXX
#define BANDIT_HXX

#include <stdio.h>
#include <iostream>
#include <vector>
using namespace std;

class Bandit{
    public:
        virtual int select_arm(double* context) = 0;
        virtual void update(double reward, double regret, int choice) = 0;
};

#endif