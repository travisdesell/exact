#include <iostream>
#include <cstdio>
#include "strategic/state.hxx"

class BanditState: public State {
    private:
        int choice;

    public:
        BanditState(int choice) {
            this->choice = choice;
        }

        void update(int new_choice) {
            this->choice = new_choice;
        }

        int get_choice() {
            return this->choice;
        }
};