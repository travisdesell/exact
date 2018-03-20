bool EXALT::insert_genome(RNN_Genome* genome) {
    float new_fitness = genome->get_best_validation_error();
    int new_generation_id = genome->get_generation_id();

    bool was_inserted = true;
    bool was_best_test_genome = false;

    inserted_genomes++;

    if (genome->get_best_validation_error() != EXALT_MAX_FLOAT) {
        write_individual_hyperparameters(genome);

        int genome_test_predictions = genome->get_test_predictions();
        int best_genome_test_predictions = 0;
        if (best_predictions_genome != NULL) {
            best_genome_test_predictions = best_predictions_genome->get_test_predictions();
        }

        if (genome_test_predictions > best_genome_test_predictions) {
            CNN_Genome *previous_best = best_predictions_genome;
 
            best_predictions_genome = genome;

#ifdef _MYSQL_
            genome->export_to_database(id);
            cout << "set new best predictions genome id to: " << genome->get_genome_id();
            best_predictions_genome_id = genome->get_genome_id();
#endif

            if (genomes.size() > 0) {
                //delete best_predictions_genome if it is not in the population
                CNN_Genome *worst = genomes.back();
                cout << "got worst" << endl;
                if (previous_best != NULL && worst->get_best_validation_error() < previous_best->get_best_validation_error()) {
                    cout << "deleting previous best" << endl;
                    delete previous_best;
                    cout << "deleted previous best" << endl;
                }
            }

            cout << "found new best predictions genome!" << endl;

            was_best_predictions_genome = true;
        }
    }

    for (auto i = generated_from_map.begin(); i != generated_from_map.end(); i++) {
        generated_from_map[i->first] += genome->get_generated_by(i->first);
    }

    cout << "genomes evaluated: " << setw(10) << inserted_genomes << ", inserting: " << parse_fitness(genome->get_best_validation_error()) << endl;

    int32_t duplicate_genome = population_contains(genome);
    if (duplicate_genome >= 0) {
        //if fitness is better, replace this genome with new one
        cout << "found duplicate at position: " << duplicate_genome << endl;

        CNN_Genome *duplicate = genomes[duplicate_genome];
        if (duplicate->get_best_validation_error() > genome->get_best_validation_error()) {
            //erase the genome with loewr fitness from the vector;
            cout << "REPLACING DUPLICATE GENOME, fitness of genome in search: " << parse_fitness(duplicate->get_best_validation_error()) << ", new fitness: " << parse_fitness(genome->get_best_validation_error()) << endl;
            genomes.erase(genomes.begin() + duplicate_genome);
            delete duplicate;

        } else {
            cerr << "\tpopulation already contains genome! not inserting." << endl;
            if (!was_best_predictions_genome) delete genome;

            if (output_directory.compare("") != 0) write_statistics(new_generation_id, new_fitness);
            return false;
        }
    }

    cout << "performing sanity check." << endl;

    if (!genome->sanity_check(SANITY_CHECK_BEFORE_INSERT)) {
        cout << "ERROR: genome " << genome->get_generation_id() << " failed sanity check before insert!" << endl;
        exit(1);
    }
    cout << "genome " << genome->get_generation_id() << " passed sanity check with fitness: " << parse_fitness(genome->get_best_validation_error()) << endl;

    if (genomes.size() == 0 || genome->get_best_validation_error() < genomes[0]->get_best_validation_error()) {
        cout << "new best fitness!" << endl;

        cout << "writing new best (data) to: " << (output_directory + "/global_best_" + to_string(inserted_genomes) + ".txt") << endl;

        genome->write_to_file(output_directory + "/global_best_" + to_string(inserted_genomes) + ".txt");

        cout << "writing new best (graphviz) to: " << (output_directory + "/global_best_" + to_string(inserted_genomes) + ".txt") << endl;

        ofstream gv_file(output_directory + "/global_best_" + to_string(inserted_genomes) + ".gv");
        gv_file << "#EXALT settings: " << endl;

        gv_file << "#EXALT settings: " << endl;

        gv_file << "#\tinitial_batch_size_min: " << initial_batch_size_min << endl;
        gv_file << "#\tinitial_batch_size_max: " << initial_batch_size_max << endl;
        gv_file << "#\tbatch_size_min: " << batch_size_min << endl;
        gv_file << "#\tbatch_size_max: " << batch_size_max << endl;

        gv_file << "#\tinitial_mu_min: " << initial_mu_min << endl;
        gv_file << "#\tinitial_mu_max: " << initial_mu_max << endl;
        gv_file << "#\tmu_min: " << mu_min << endl;
        gv_file << "#\tmu_max: " << mu_max << endl;

        gv_file << "#\tinitial_mu_delta_min: " << initial_mu_delta_min << endl;
        gv_file << "#\tinitial_mu_delta_max: " << initial_mu_delta_max << endl;
        gv_file << "#\tmu_delta_min: " << mu_delta_min << endl;
        gv_file << "#\tmu_delta_max: " << mu_delta_max << endl;

        gv_file << "#\tinitial_learning_rate_min: " << initial_learning_rate_min << endl;
        gv_file << "#\tinitial_learning_rate_max: " << initial_learning_rate_max << endl;
        gv_file << "#\tlearning_rate_min: " << learning_rate_min << endl;
        gv_file << "#\tlearning_rate_max: " << learning_rate_max << endl;

        gv_file << "#\tinitial_learning_rate_delta_min: " << initial_learning_rate_delta_min << endl;
        gv_file << "#\tinitial_learning_rate_delta_max: " << initial_learning_rate_delta_max << endl;
        gv_file << "#\tlearning_rate_delta_min: " << learning_rate_delta_min << endl;
        gv_file << "#\tlearning_rate_delta_max: " << learning_rate_delta_max << endl;

        gv_file << "#\tinitial_weight_decay_min: " << initial_weight_decay_min << endl;
        gv_file << "#\tinitial_weight_decay_max: " << initial_weight_decay_max << endl;
        gv_file << "#\tweight_decay_min: " << weight_decay_min << endl;
        gv_file << "#\tweight_decay_max: " << weight_decay_max << endl;

        gv_file << "#\tinitial_weight_decay_delta_min: " << initial_weight_decay_delta_min << endl;
        gv_file << "#\tinitial_weight_decay_delta_max: " << initial_weight_decay_delta_max << endl;
        gv_file << "#\tweight_decay_delta_min: " << weight_decay_delta_min << endl;
        gv_file << "#\tweight_decay_delta_max: " << weight_decay_delta_max << endl;

        gv_file << "#\tepsilon: " << epsilon << endl;

        gv_file << "#\tinitial_alpha_min: " << initial_alpha_min << endl;
        gv_file << "#\tinitial_alpha_max: " << initial_alpha_max << endl;
        gv_file << "#\talpha_min: " << alpha_min << endl;
        gv_file << "#\talpha_max: " << alpha_max << endl;

        gv_file << "#\tinitial_velocity_reset_min: " << initial_velocity_reset_min << endl;
        gv_file << "#\tinitial_velocity_reset_max: " << initial_velocity_reset_max << endl;
        gv_file << "#\tvelocity_reset_min: " << velocity_reset_min << endl;
        gv_file << "#\tvelocity_reset_max: " << velocity_reset_max << endl;

        gv_file << "#\tinitial_input_dropout_probability_min: " << initial_input_dropout_probability_min << endl;
        gv_file << "#\tinitial_input_dropout_probability_max: " << initial_input_dropout_probability_max << endl;
        gv_file << "#\tinput_dropout_probability_min: " << input_dropout_probability_min << endl;
        gv_file << "#\tinput_dropout_probability_max: " << input_dropout_probability_max << endl;

        gv_file << "#\tinitial_hidden_dropout_probability_min: " << initial_hidden_dropout_probability_min << endl;
        gv_file << "#\tinitial_hidden_dropout_probability_max: " << initial_hidden_dropout_probability_max << endl;
        gv_file << "#\thidden_dropout_probability_min: " << hidden_dropout_probability_min << endl;
        gv_file << "#\thidden_dropout_probability_max: " << hidden_dropout_probability_max << endl;

        gv_file << "#\tmax_epochs: " << max_epochs << endl;
        gv_file << "#\treset_weights_chance: " << reset_weights_chance << endl;

        gv_file << "#\tcrossover_settings: " << endl;
        gv_file << "#\t\tcrossover_rate: " << crossover_rate << endl;
        gv_file << "#\t\tmore_fit_parent_crossover: " << more_fit_parent_crossover << endl;
        gv_file << "#\t\tless_fit_parent_crossover: " << less_fit_parent_crossover << endl;
        gv_file << "#\t\tcrossover_alter_edge_type: " << crossover_alter_edge_type << endl;

        gv_file << "#\tmutation_settings: " << endl;
        gv_file << "#\t\tnumber_mutations: " << number_mutations << endl;
        gv_file << "#\t\tedge_alter_type: " << edge_alter_type << endl;
        gv_file << "#\t\tedge_disable: " << edge_disable << endl;
        gv_file << "#\t\tedge_split: " << edge_split << endl;
        gv_file << "#\t\tedge_add: " << edge_add << endl;
        gv_file << "#\t\tnode_change_size: " << node_change_size << endl;
        gv_file << "#\t\tnode_change_size_x: " << node_change_size_x << endl;
        gv_file << "#\t\tnode_change_size_y: " << node_change_size_y << endl;
        gv_file << "#\t\tnode_add: " << node_add << endl;
        gv_file << "#\t\tnode_split: " << node_split << endl;
        gv_file << "#\t\tnode_merge: " << node_merge << endl;
        gv_file << "#\t\tnode_enable: " << node_enable << endl;
        gv_file << "#\t\tnode_disable: " << node_disable << endl;

        genome->print_graphviz(gv_file);
        gv_file.close();
    }
    cout << endl;

    if (genomes.size() == 0) {
        cout << "checking if individual should be inserted or not, genomes.size(): " << genomes.size() << ", population_size: " << population_size << ", genome->get_best_validation_error(): " << genome->get_best_validation_error() << ", genomes is empty!" << endl;
    } else {
        cout << "checking if individual should be inserted or not, genomes.size(): " << genomes.size() << ", population_size: " << population_size << ", genome->get_best_validation_error(): " << genome->get_best_validation_error() << ", genomes.back()->get_best_validation_error(): " << genomes.back()->get_best_validation_error() << endl;
    }

    if ((int32_t)genomes.size() >= population_size && genome->get_best_validation_error() >= genomes.back()->get_best_validation_error()) {
        //this will not be inserted into the population
        cout << "not inserting genome due to poor fitness" << endl;
        was_inserted = false;

        if (!was_best_predictions_genome) delete genome;
    } else {
        cout << "updating search statistics" << endl;

        for (auto i = inserted_from_map.begin(); i != inserted_from_map.end(); i++) {
            inserted_from_map[i->first] += genome->get_generated_by(i->first);
        }

        cout << "inserting new genome" << endl;
        //inorder insert the new individual
        genomes.insert( upper_bound(genomes.begin(), genomes.end(), genome, sort_genomes_by_validation_error()), genome);

        cout << "inserted the new genome" << endl;

        //delete the worst individual if we've reached the population size
        if ((int32_t)genomes.size() > population_size) {
            cout << "deleting worst genome" << endl;
            CNN_Genome *worst = genomes.back();
            genomes.pop_back();

            if (worst->get_genome_id() != best_predictions_genome_id) {
                delete worst;
            }
        }
    }

    cout << "genome fitnesses:" << endl;
    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        cout << "\t" << setw(4) << i << " -- genome: " << setw(10) << genomes[i]->get_generation_id() << ", "
            << setw(10) << left << "val err: " << right << setw(12) << setprecision(2) << fixed << parse_fitness(genomes[i]->get_best_validation_error())
            << " (" << setw(5) << fixed << setprecision(2) << genomes[i]->get_best_validation_rate() << "%)"
            << ", " << setw(10) << left << "test err: " << right << setw(12) << setprecision(2) << fixed << parse_fitness(genomes[i]->get_test_error())
            << " (" << setw(5) << fixed << setprecision(2) << genomes[i]->get_test_rate() << "%), "
            << setw(10) << left << "train err: " << right << setw(12) << setprecision(2) << fixed << parse_fitness(genomes[i]->get_training_error())
            << " (" << setw(5) << fixed << setprecision(2) << genomes[i]->get_training_rate() << "%) on ep: " << genomes[i]->get_best_epoch() 
            //<< ", reachable edges: " << genomes[i]->get_number_reachable_edges()
            //<< ", reachable nodes: " << genomes[i]->get_number_reachable_nodes()
            << ", mu: " << setw(8) << fixed << setprecision(5) << genomes[i]->get_initial_mu()
            << ", mu_d: " << setw(8) << fixed << setprecision(5) << genomes[i]->get_mu_delta()
            << ", lr: " << setw(8) << fixed << setprecision(5) << genomes[i]->get_initial_learning_rate()
            << ", lr_d: " << setw(8) << fixed << setprecision(5) << genomes[i]->get_learning_rate_delta()
            << ", wd: " << setw(8) << fixed << setprecision(5) << genomes[i]->get_initial_weight_decay()
            << ", wd_d: " << setw(8) << fixed << setprecision(5) << genomes[i]->get_weight_decay_delta()
            << ", a: " << setw(8) << fixed << setprecision(5) << genomes[i]->get_alpha()
            << ", vr: " << setw(6) << fixed << setprecision(5) << genomes[i]->get_velocity_reset()
            << ", id: " << setw(8) << fixed << setprecision(5) << genomes[i]->get_input_dropout_probability()
            << ", hd: " << setw(8) << fixed << setprecision(5) << genomes[i]->get_hidden_dropout_probability()
            << ", bs: " << setw(6) << fixed << setprecision(5) << genomes[i]->get_batch_size()
            << endl;
    }

    if (best_predictions_genome != NULL) {
        cout << "best predictions genome validation predictions: " << best_predictions_genome->get_best_validation_predictions() << ", training predictions: " << best_predictions_genome->get_training_predictions() << ", test predictions: " << best_predictions_genome->get_test_predictions() << ", validation error: " << best_predictions_genome->get_best_validation_error() << endl;
    }

    /*
    cout << "genome best error: " << endl;
    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        cout << "\t" << setw(4) << i << " -- genome: " << setw(10) << genomes[i]->get_generation_id() << ", ";
        genomes[i]->print_best_error(cout);
    }

    cout << "genome correct predictions: " << endl;
    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        cout << "\t" << setw(4) << i << " -- genome: " << setw(10) << genomes[i]->get_generation_id() << ", ";
        genomes[i]->print_best_predictions(cout);
    }
    */

    cout << endl;

    if (output_directory.compare("") != 0) write_statistics(new_generation_id, new_fitness);
    return was_inserted;
}


