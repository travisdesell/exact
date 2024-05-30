#ifndef STRATEGY_STATE_HXX
#define STRATEGY_STATE_HXX

class State {
    /**
     * State is an empty class because depending on the task the state of the
     * strategy can be completely different.
     */
    
    public:
        /**
         * A default destructor for this abstract class.
         */
        virtual ~State() = default;
};


#endif
