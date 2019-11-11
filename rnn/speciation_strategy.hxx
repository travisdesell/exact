#ifndef EXAMM_SPECIATION_STRATEGY_HXX
#define EXAMM_SPECIATION_STRATEGY_HXX


class SpeciationStrategy {
    
    public:
        /**
         * Gets the fitness of the best genome of all the islands
         * \return the best fitness over all islands
         */
        virtual double get_best_fitness() = 0;

        /**
         * Gets the fitness of the worst genome of all the islands
         * \return the worst fitness over all islands
         */
        virtual double get_worst_fitness() = 0;

        /**
         * Gets the best genome of all the islands
         * \return the best genome of all islands
         */
        virtual RNN_Genome* get_best_genome() = 0;

        /**
         * Gets the the worst genome of all the islands
         * \return the worst genome of all islands
         */
        virtual RNN_Genome* get_worst_genome() = 0;


        /**
         * Prints out all the island's populations
         */
        virtual void print_population() = 0;

        /**
         * Inserts a <b>copy</b> of the genome into this speciation strategy.
         *
         * The caller of this method will need to free the memory of the genome passed
         * into this method.
         *
         * \param genome is the genome to insert.
         * \return a value < 0 if the genome was not inserted, 0 if it was a new best genome
         * or > 0 otherwise.
         */
        virtual int32_t insert_genome(RNN_Genome* genome) = 0;


        /**
         * Generates a new genome.
         *
         * \return the newly generated genome.
         */
        RNN_Genome* generate_genome();
};


#endif

