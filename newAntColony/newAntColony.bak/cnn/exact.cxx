#include <algorithm>
using std::sort;
using std::upper_bound;

#include <chrono>

#include <iomanip>
using std::fixed;
using std::setprecision;
using std::setw;
using std::left;
using std::right;

#include <iostream>
using std::fstream;
using std::ostream;
using std::istream;

#include <limits>
using std::numeric_limits;

#include <random>
using std::minstd_rand0;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

#include <sstream>
using std::istringstream;
using std::ostringstream;

#include <string>
using std::to_string;

#include <vector>
using std::vector;

#include "image_tools/image_set.hxx"
#include "comparison.hxx"
#include "cnn_node.hxx"
#include "cnn_edge.hxx"
#include "cnn_genome.hxx"
#include "exact.hxx"

#ifdef _MYSQL_
#include "common/db_conn.hxx"
#endif

#include "stdlib.h"

#ifdef _MYSQL_

bool EXACT::exists_in_database(int exact_id) {
    ostringstream query;

    query << "SELECT * FROM exact_search WHERE id = " << exact_id;

    mysql_exact_query(query.str());
    
    MYSQL_RES *result = mysql_store_result(exact_db_conn);
    bool found;
    if (result != NULL) {
        if (mysql_num_rows(result) > 0) {
            found = true;
        } else {
            found = false;
        }

    } else {
        cerr << "ERROR in mysql query: '" << query.str() << "'" << endl;
        exit(1);
    }
    mysql_free_result(result);

    return found;
}

EXACT::EXACT(int exact_id) {
    ostringstream query;

    query << "SELECT * FROM exact_search WHERE id = " << exact_id;

    mysql_exact_query(query.str());
    
    MYSQL_RES *result = mysql_store_result(exact_db_conn);
    if (result != NULL) {
        MYSQL_ROW row = mysql_fetch_row(result);

        id = exact_id;  //is row 0

        int column = 0;

        search_name = string(row[++column]);
        output_directory = string(row[++column]);
        training_filename = string(row[++column]);
        validation_filename = string(row[++column]);
        test_filename = string(row[++column]);

        number_training_images = atoi(row[++column]);
        number_validation_images = atoi(row[++column]);
        number_test_images = atoi(row[++column]);

        padding = atoi(row[++column]);

        image_channels = atoi(row[++column]);
        image_rows = atoi(row[++column]);
        image_cols = atoi(row[++column]);
        number_classes = atoi(row[++column]);

        population_size = atoi(row[++column]);
        node_innovation_count = atoi(row[++column]);
        edge_innovation_count = atoi(row[++column]);

        best_predictions_genome_id = atoi(row[++column]);

        genomes_generated = atoi(row[++column]);
        inserted_genomes = atoi(row[++column]);
        max_genomes = atoi(row[++column]);

        reset_weights = atoi(row[++column]);
        max_epochs = atoi(row[++column]);
        use_sfmp = atoi(row[++column]);
        use_node_operations = atoi(row[++column]);

        initial_batch_size_min = atoi(row[++column]);
        initial_batch_size_max = atoi(row[++column]);
        batch_size_min = atoi(row[++column]);
        batch_size_max = atoi(row[++column]);

        initial_mu_min = atof(row[++column]);
        initial_mu_max = atof(row[++column]);
        mu_min = atof(row[++column]);
        mu_max = atof(row[++column]);

        initial_mu_delta_min = atof(row[++column]);
        initial_mu_delta_max = atof(row[++column]);
        mu_delta_min = atof(row[++column]);
        mu_delta_max = atof(row[++column]);

        initial_learning_rate_min = atof(row[++column]);
        initial_learning_rate_max = atof(row[++column]);
        learning_rate_min = atof(row[++column]);
        learning_rate_max = atof(row[++column]);

        initial_learning_rate_delta_min = atof(row[++column]);
        initial_learning_rate_delta_max = atof(row[++column]);
        learning_rate_delta_min = atof(row[++column]);
        learning_rate_delta_max = atof(row[++column]);

        initial_weight_decay_min = atof(row[++column]);
        initial_weight_decay_max = atof(row[++column]);
        weight_decay_min = atof(row[++column]);
        weight_decay_max = atof(row[++column]);

        initial_weight_decay_delta_min = atof(row[++column]);
        initial_weight_decay_delta_max = atof(row[++column]);
        weight_decay_delta_min = atof(row[++column]);
        weight_decay_delta_max = atof(row[++column]);

        epsilon = atof(row[++column]);

        initial_alpha_min = atof(row[++column]);
        initial_alpha_max = atof(row[++column]);
        alpha_min = atof(row[++column]);
        alpha_max = atof(row[++column]);

        initial_velocity_reset_min = atoi(row[++column]);
        initial_velocity_reset_max = atoi(row[++column]);
        velocity_reset_min = atoi(row[++column]);
        velocity_reset_max = atoi(row[++column]);

        initial_input_dropout_probability_min = atof(row[++column]);
        initial_input_dropout_probability_max = atof(row[++column]);
        input_dropout_probability_min = atof(row[++column]);
        input_dropout_probability_max = atof(row[++column]);

        initial_hidden_dropout_probability_min = atof(row[++column]);
        initial_hidden_dropout_probability_max = atof(row[++column]);
        hidden_dropout_probability_min = atof(row[++column]);
        hidden_dropout_probability_max = atof(row[++column]);

        reset_weights_chance = atof(row[++column]);

        crossover_rate = atof(row[++column]);
        more_fit_parent_crossover = atof(row[++column]);
        less_fit_parent_crossover = atof(row[++column]);
        crossover_alter_edge_type = atof(row[++column]);

        number_mutations = atoi(row[++column]);
        edge_alter_type = atof(row[++column]);
        edge_disable = atof(row[++column]);
        edge_enable = atof(row[++column]);
        edge_split = atof(row[++column]);
        edge_add = atof(row[++column]);
        node_change_size = atof(row[++column]);
        node_change_size_x = atof(row[++column]);
        node_change_size_y = atof(row[++column]);
        node_add = atof(row[++column]);
        node_split = atof(row[++column]);
        node_merge = atof(row[++column]);
        node_enable = atof(row[++column]);
        node_disable = atof(row[++column]);

        istringstream generator_iss(row[++column]);
        generator_iss >> generator;
        //cout << "read generator from database: " << generator << endl;

        istringstream normal_distribution_iss(row[++column]);
        normal_distribution_iss >> normal_distribution;
        //cout << "read normal_distribution from database: " << normal_distribution << endl;

        istringstream rng_long_iss(row[++column]);
        rng_long_iss >> rng_long;
        //cout << "read rng_long from database: " << rng_long << endl;

        istringstream rng_float_iss(row[++column]);
        rng_float_iss >> rng_float;
        //cout << "read rng_float from database: " << rng_float << endl;

        istringstream inserted_from_map_iss(row[++column]);
        read_map(inserted_from_map_iss, inserted_from_map);

        istringstream generated_from_map_iss(row[++column]);
        read_map(generated_from_map_iss, generated_from_map);

        ostringstream genome_query;
        genome_query << "SELECT id FROM cnn_genome WHERE exact_id = " << id << " ORDER BY best_validation_error LIMIT " << population_size;
        //cout << genome_query.str() << endl;

        mysql_exact_query(genome_query.str());

        MYSQL_RES *genome_result = mysql_store_result(exact_db_conn);

        //cout << "got genome result" << endl;

        MYSQL_ROW genome_row;
        while ((genome_row = mysql_fetch_row(genome_result)) != NULL) {
            int genome_id = atoi(genome_row[0]);
            //cout << "got genome with id: " << genome_id << endl;

            CNN_Genome *genome = new CNN_Genome(genome_id);
            genomes.insert( upper_bound(genomes.begin(), genomes.end(), genome, sort_genomes_by_validation_error()), genome);
        }

        cout << "got " << genomes.size() << " genomes." << endl;
        cout << "population_size: " << population_size << endl;
        cout << "inserted_genomes: " << inserted_genomes << ", max_genomes: " << max_genomes << endl;

        mysql_free_result(result);

        if (best_predictions_genome_id > 0) {
            best_predictions_genome = new CNN_Genome(best_predictions_genome_id);
        } else {
            best_predictions_genome = NULL;
        }
    } else {
        cerr << "ERROR! could not find exact_search in database with id: " << exact_id << endl;
        exit(1);
    }
}

void EXACT::export_to_database() {
    ostringstream query;
    if (id >= 0) {
        query << "REPLACE INTO exact_search SET id = " << id << ",";
    } else {
        query << "INSERT INTO exact_search SET";
    }

    //cout << "exporting exact to database!" << endl;

    query << " search_name = '" << search_name << "'"
        << ", output_directory = '" << output_directory << "'"
        << ", training_filename = '" << training_filename << "'"
        << ", validation_filename = '" << validation_filename << "'"
        << ", test_filename = '" << test_filename << "'"

        << ", number_training_images = " << number_training_images
        << ", number_validation_images = " << number_validation_images
        << ", number_test_images = " << number_test_images

        << ", padding = " << padding

        << ", image_channels = " << image_channels
        << ", image_rows = " << image_rows
        << ", image_cols = " << image_cols
        << ", number_classes = " << number_classes

        << ", population_size = " << population_size
        << ", node_innovation_count = " << node_innovation_count
        << ", edge_innovation_count = " << edge_innovation_count

        << ", best_predictions_genome_id = " << best_predictions_genome_id

        << ", genomes_generated = " << genomes_generated
        << ", inserted_genomes = " << inserted_genomes
        << ", max_genomes = " << max_genomes

        << ", reset_weights = " << reset_weights
        << ", max_epochs = " << max_epochs
        << ", use_sfmp = " << use_sfmp
        << ", use_node_operations = " << use_node_operations

        << ", initial_batch_size_min = " << initial_batch_size_min
        << ", initial_batch_size_max = " << initial_batch_size_max
        << ", batch_size_min = " << batch_size_min
        << ", batch_size_max = " << batch_size_max

        << ", initial_mu_min = " << initial_mu_min
        << ", initial_mu_max = " << initial_mu_max
        << ", mu_min = " << mu_min
        << ", mu_max = " << mu_max

        << ", initial_mu_delta_min = " << initial_mu_delta_min
        << ", initial_mu_delta_max = " << initial_mu_delta_max
        << ", mu_delta_min = " << mu_delta_min
        << ", mu_delta_max = " << mu_delta_max

        << ", initial_learning_rate_min = " << initial_learning_rate_min
        << ", initial_learning_rate_max = " << initial_learning_rate_max
        << ", learning_rate_min = " << learning_rate_min
        << ", learning_rate_max = " << learning_rate_max

        << ", initial_learning_rate_delta_min = " << initial_learning_rate_delta_min
        << ", initial_learning_rate_delta_max = " << initial_learning_rate_delta_max
        << ", learning_rate_delta_min = " << learning_rate_delta_min
        << ", learning_rate_delta_max = " << learning_rate_delta_max

        << ", initial_weight_decay_min = " << initial_weight_decay_min
        << ", initial_weight_decay_max = " << initial_weight_decay_max
        << ", weight_decay_min = " << weight_decay_min
        << ", weight_decay_max = " << weight_decay_max

        << ", initial_weight_decay_delta_min = " << initial_weight_decay_delta_min
        << ", initial_weight_decay_delta_max = " << initial_weight_decay_delta_max
        << ", weight_decay_delta_min = " << weight_decay_delta_min
        << ", weight_decay_delta_max = " << weight_decay_delta_max

        << ", epsilon = " << epsilon

        << ", initial_alpha_min = " << initial_alpha_min
        << ", initial_alpha_max = " << initial_alpha_max
        << ", alpha_min = " << alpha_min
        << ", alpha_max = " << alpha_max

        << ", initial_velocity_reset_min = " << initial_velocity_reset_min
        << ", initial_velocity_reset_max = " << initial_velocity_reset_max
        << ", velocity_reset_min = " << velocity_reset_min
        << ", velocity_reset_max = " << velocity_reset_max

        << ", initial_input_dropout_probability_min = " << initial_input_dropout_probability_min
        << ", initial_input_dropout_probability_max = " << initial_input_dropout_probability_max
        << ", input_dropout_probability_min = " << input_dropout_probability_min
        << ", input_dropout_probability_max = " << input_dropout_probability_max

        << ", initial_hidden_dropout_probability_min = " << initial_hidden_dropout_probability_min
        << ", initial_hidden_dropout_probability_max = " << initial_hidden_dropout_probability_max
        << ", hidden_dropout_probability_min = " << hidden_dropout_probability_min
        << ", hidden_dropout_probability_max = " << hidden_dropout_probability_max

        << ", reset_weights_chance = " << reset_weights_chance

        << ", crossover_rate = " << crossover_rate
        << ", more_fit_parent_crossover = " << more_fit_parent_crossover
        << ", less_fit_parent_crossover = " << less_fit_parent_crossover
        << ", crossover_alter_edge_type = " << crossover_alter_edge_type

        << ", number_mutations = " << number_mutations
        << ", edge_alter_type = " << edge_alter_type
        << ", edge_disable = " << edge_disable
        << ", edge_enable = " << edge_enable
        << ", edge_split = " << edge_split
        << ", edge_add = " << edge_add
        << ", node_change_size = " << node_change_size
        << ", node_change_size_x = " << node_change_size_x
        << ", node_change_size_y = " << node_change_size_y
        << ", node_add = " << node_add
        << ", node_split = " << node_split
        << ", node_merge = " << node_merge
        << ", node_enable = " << node_enable
        << ", node_disable = " << node_disable

        << ", generator = '" << generator << "'"
        << ", normal_distribution = '" << normal_distribution << "'"
        << ", rng_long = '" << rng_long << "'"
        << ", rng_float = '" << rng_float << "'"

        << ", inserted_from_map = '";

    write_map(query, inserted_from_map);

    query << "'"
        << ", generated_from_map = '";
    write_map(query, generated_from_map);
    query << "'";

    cout << query.str() << endl;
    mysql_exact_query(query.str());

    if (id < 0) {
        id = mysql_exact_last_insert_id();
        cout << "inserted EXACT search with id: " << id << endl;
    }

    //need to insert genomes
    for (uint32_t i = 0; i < genomes.size(); i++) {
        genomes[i]->export_to_database(id);
    }

    if ((int32_t)genomes.size() == population_size) {
        ostringstream delete_query;
        delete_query << "DELETE FROM cnn_genome WHERE exact_id = " << id << " AND ";
        delete_query << "(";

        for (uint32_t i = 0; i < genomes.size(); i++) {
            delete_query << "id != " << genomes[i]->get_genome_id();

            if (i < (genomes.size() - 1)) delete_query << " AND ";
        }

        if (best_predictions_genome_id > 0) {
            delete_query << " AND id != " << best_predictions_genome_id;
        }

        delete_query << ")";

        cout << delete_query.str() << endl;
        mysql_exact_query(delete_query.str());

        ostringstream delete_node_query;
        delete_node_query << "DELETE FROM cnn_node WHERE exact_id = " << id << " AND genome_id > 0 AND NOT EXISTS(SELECT id FROM cnn_genome WHERE cnn_genome.id = cnn_node.genome_id)";
        cout <<  delete_node_query.str() << endl;
        mysql_exact_query(delete_node_query.str());

        ostringstream delete_edge_query;
        delete_edge_query << "DELETE FROM cnn_edge WHERE exact_id = " << id << " AND genome_id > 0 AND NOT EXISTS(SELECT id FROM cnn_genome WHERE cnn_genome.id = cnn_edge.genome_id)";
        cout <<  delete_edge_query.str() << endl;
        mysql_exact_query(delete_edge_query.str());
    }
}

void EXACT::update_database() {
    if (id < 0) {
        cerr << "ERROR: Cannot update an exact search in the databse if it has not already been entered, id was < 0." << endl;
        return;
    }

    ostringstream query;
    query << "UPDATE exact_search SET";

    query << " genomes_generated = " << genomes_generated
        << ", inserted_genomes = " << inserted_genomes

        << ", node_innovation_count = " << node_innovation_count
        << ", edge_innovation_count = " << edge_innovation_count
        << ", best_predictions_genome_id = " << best_predictions_genome_id
        << ", inserted_from_map = '";

    write_map(query, inserted_from_map);

    query << "'"
        << ", generated_from_map = '";
    write_map(query, generated_from_map);

    query << "'"
        << ", generator = '" << generator << "'"
        << ", normal_distribution = '" << normal_distribution << "'"
        << ", rng_long = '" << rng_long << "'"
        << ", rng_float = '" << rng_float << "'"
        << " WHERE id = " << id;

    cout << query.str() << endl;
    mysql_exact_query(query.str());

    //genomes are inserted separately

    if ((int32_t)genomes.size() == population_size) {
        ostringstream delete_query;
        delete_query << "DELETE FROM cnn_genome WHERE exact_id = " << id << " AND ";
        delete_query << "(";

        for (uint32_t i = 0; i < genomes.size(); i++) {
            delete_query << "id != " << genomes[i]->get_genome_id();

            if (i < (genomes.size() - 1)) delete_query << " AND ";
        }

        if (best_predictions_genome_id > 0) {
            delete_query << " AND id != " << best_predictions_genome_id;
        }

        delete_query << ")";

        cout << delete_query.str() << endl;
        mysql_exact_query(delete_query.str());

        ostringstream delete_node_query;
        delete_node_query << "DELETE FROM cnn_node WHERE exact_id = " << id << " AND genome_id > 0 AND NOT EXISTS(SELECT id FROM cnn_genome WHERE cnn_genome.id = cnn_node.genome_id)";
        cout <<  delete_node_query.str() << endl;
        mysql_exact_query(delete_node_query.str());

        ostringstream delete_edge_query;
        delete_edge_query << "DELETE FROM cnn_edge WHERE exact_id = " << id << " AND genome_id > 0 AND NOT EXISTS(SELECT id FROM cnn_genome WHERE cnn_genome.id = cnn_edge.genome_id)";
        cout <<  delete_edge_query.str() << endl;
        mysql_exact_query(delete_edge_query.str());
    }
}

#endif

EXACT::EXACT(const ImagesInterface &training_images, const ImagesInterface &validation_images, const ImagesInterface &test_images, int _padding, int _population_size, int _max_epochs, bool _use_sfmp, bool _use_node_operations, int _max_genomes, string _output_directory, string _search_name, bool _reset_weights) {
    id = -1;

    search_name = _search_name;
    output_directory = _output_directory;

    training_filename = training_images.get_filename(); 
    validation_filename = validation_images.get_filename();
    test_filename = test_images.get_filename();

    reset_weights = _reset_weights;
    max_epochs = _max_epochs;
    use_sfmp = _use_sfmp;
    use_node_operations = _use_node_operations;

    max_genomes = _max_genomes;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    //unsigned seed = 10;

    generator = minstd_rand0(seed);
    rng_long = uniform_int_distribution<long>(-numeric_limits<long>::max(), numeric_limits<long>::max());
    rng_float = uniform_real_distribution<float>(0, 1.0);

    node_innovation_count = 0;
    edge_innovation_count = 0;

    best_predictions_genome_id = -1;
    best_predictions_genome = NULL;

    inserted_genomes = 0;

    population_size = _population_size;

    number_training_images = training_images.get_number_images();
    number_validation_images = validation_images.get_number_images();
    number_test_images = test_images.get_number_images();

    padding = _padding;

    if (training_images.get_image_channels() != test_images.get_image_channels()) {
        cerr << "ERROR, could not start EXACT search because number training channels != number test channels in images" << endl;
        exit(1);
    }

    if (training_images.get_image_height() != test_images.get_image_height()) {
        cerr << "ERROR, could not start EXACT search because number training rows != number test rows in images" << endl;
        exit(1);
    }

    if (training_images.get_image_width() != test_images.get_image_width()) {
        cerr << "ERROR, could not start EXACT search because number training cols != number test cols in images" << endl;
        exit(1);
    }

    if (training_images.get_number_classes() != test_images.get_number_classes()) {
        cerr << "ERROR, could not start EXACT search because number training classes != number test classes in images" << endl;
        exit(1);
    }

    if (training_images.get_image_channels() != validation_images.get_image_channels()) {
        cerr << "ERROR, could not start EXACT search because number training channels != number validation channels in images" << endl;
        exit(1);
    }

    if (training_images.get_image_height() != validation_images.get_image_height()) {
        cerr << "ERROR, could not start EXACT search because number training rows != number validation rows in images" << endl;
        exit(1);
    }

    if (training_images.get_image_width() != validation_images.get_image_width()) {
        cerr << "ERROR, could not start EXACT search because number training cols != number validation cols in images" << endl;
        exit(1);
    }

    if (training_images.get_number_classes() != validation_images.get_number_classes()) {
        cerr << "ERROR, could not start EXACT search because number training classes != number validation classes in images" << endl;
        exit(1);
    }


    image_channels = training_images.get_image_channels();
    image_rows = training_images.get_image_height();
    image_cols = training_images.get_image_width();
    number_classes = training_images.get_number_classes();

    inserted_from_map["disable_edge"] = 0;
    inserted_from_map["enable_edge"] = 0;
    inserted_from_map["split_edge"] = 0;
    inserted_from_map["add_edge"] = 0;
    inserted_from_map["alter_edge_type"] = 0;
    inserted_from_map["change_size"] = 0;
    inserted_from_map["change_size_x"] = 0;
    inserted_from_map["change_size_y"] = 0;
    inserted_from_map["crossover"] = 0;
    inserted_from_map["reset_weights"] = 0;
    inserted_from_map["add_node"] = 0;
    inserted_from_map["split_node"] = 0;
    inserted_from_map["merge_node"] = 0;
    inserted_from_map["enable_node"] = 0;
    inserted_from_map["disable_node"] = 0;

    generated_from_map["disable_edge"] = 0;
    generated_from_map["enable_edge"] = 0;
    generated_from_map["split_edge"] = 0;
    generated_from_map["add_edge"] = 0;
    generated_from_map["alter_edge_type"] = 0;
    generated_from_map["change_size"] = 0;
    generated_from_map["change_size_x"] = 0;
    generated_from_map["change_size_y"] = 0;
    generated_from_map["crossover"] = 0;
    generated_from_map["reset_weights"] = 0;
    generated_from_map["add_node"] = 0;
    generated_from_map["split_node"] = 0;
    generated_from_map["merge_node"] = 0;
    generated_from_map["enable_node"] = 0;
    generated_from_map["disable_node"] = 0;

    genomes_generated = 0;

    epsilon = 1.0e-7;

    initial_batch_size_min = 50;
    initial_batch_size_max = 50;
    batch_size_min = 50;
    batch_size_max = 50;

    initial_mu_min = 0.50;
    initial_mu_max = 0.50;
    mu_min = 0.50;
    mu_max = 0.50;

    initial_mu_delta_min = 0.95;
    initial_mu_delta_max = 0.95;
    mu_delta_min = 0.95;
    mu_delta_max = 0.95;

    initial_learning_rate_min = 0.0125;
    initial_learning_rate_max = 0.0125;
    learning_rate_min = 0.0125;
    learning_rate_max = 0.0125;

    initial_learning_rate_delta_min = 0.95;
    initial_learning_rate_delta_max = 0.95;
    learning_rate_delta_min = 0.95;
    learning_rate_delta_max = 0.95;

    initial_weight_decay_min = 0.0005;
    initial_weight_decay_max = 0.0005;
    weight_decay_min = 0.0005;
    weight_decay_max = 0.0005;

    initial_weight_decay_delta_min = 0.95;
    initial_weight_decay_delta_max = 0.95;
    weight_decay_delta_min = 0.95;
    weight_decay_delta_max = 0.95;

    initial_alpha_min = 0.1;
    initial_alpha_max = 0.1;
    alpha_min = 0.1;
    alpha_max = 0.1;

    initial_velocity_reset_min = 1000;
    initial_velocity_reset_max = 1000;
    velocity_reset_min = 1000;
    velocity_reset_max = 1000;

    initial_input_dropout_probability_min = 0.0000;
    initial_input_dropout_probability_max = 0.000;
    input_dropout_probability_min = 0.0000;
    input_dropout_probability_max = 0.0;

    initial_hidden_dropout_probability_min = 0.00;
    initial_hidden_dropout_probability_max = 0.00;
    hidden_dropout_probability_min = 0.0;
    hidden_dropout_probability_max = 0.0;

    /*
    initial_batch_size_min = 25;
    initial_batch_size_max = 150;
    batch_size_min = 25;
    batch_size_max = 300;

    initial_mu_min = 0.40;
    initial_mu_max = 0.60;
    mu_min = 0.0;
    mu_max = 0.99;

    initial_mu_delta_min = 0.90;
    initial_mu_delta_max = 0.99;
    mu_delta_min = 0.0;
    mu_delta_max = 1.00;

    initial_learning_rate_min = 0.001;
    initial_learning_rate_max = 0.10;
    learning_rate_min = 0.00001;
    learning_rate_max = 0.3;

    initial_learning_rate_delta_min = 0.90;
    initial_learning_rate_delta_max = 0.99;
    learning_rate_delta_min = 0.00000001;
    learning_rate_delta_max = 1.0;

    initial_weight_decay_min = 0.0001;
    initial_weight_decay_max = 0.001;
    weight_decay_min = 0.00000000;
    weight_decay_max = 0.1;

    initial_weight_decay_delta_min = 0.90;
    initial_weight_decay_delta_max = 0.99;
    weight_decay_delta_min = 0.00000001;
    weight_decay_delta_max = 1.0;

    initial_alpha_min = 0.001;
    initial_alpha_max = 0.2;
    alpha_min = 0.0001;
    alpha_max = 0.5;

    initial_velocity_reset_min = 500;
    initial_velocity_reset_max = 3000;
    velocity_reset_min = 0;
    velocity_reset_max = 60000;

    initial_input_dropout_probability_min = 0.0005;
    initial_input_dropout_probability_max = 0.002;
    input_dropout_probability_min = 0.0001;
    input_dropout_probability_max = 0.5;

    initial_hidden_dropout_probability_min = 0.05;
    initial_hidden_dropout_probability_max = 0.15;
    hidden_dropout_probability_min = 0.0;
    hidden_dropout_probability_max = 0.9;
    */


    crossover_rate = 0.20;
    more_fit_parent_crossover = 0.80;
    less_fit_parent_crossover = 0.40;
    crossover_alter_edge_type = 0.00;

    reset_weights_chance = 0.20;

    number_mutations = 1;
    if (use_sfmp) {
        edge_alter_type = 2.0;
    } else {
        edge_alter_type = 0.0;
    }

    edge_disable = 2.5;
    edge_enable = 2.5;
    edge_split = 3.0;
    edge_add = 3.0;
    node_change_size = 2.0;
    node_change_size_x = 1.0;
    node_change_size_y = 1.0;

    if (use_node_operations) {
        node_add = 3.0;
        node_split = 2.0;
        node_merge = 2.0;
        node_disable = 1.5;
        node_enable = 1.5;
    } else {
        node_add = 0.0;
        node_split = 0.0;
        node_merge = 0.0;
        node_disable = 0.0;
        node_enable = 0.0;
    }

    cout << "EXACT settings: " << endl;

    cout << "\tinitial_batch_size_min: " << initial_batch_size_min << endl;
    cout << "\tinitial_batch_size_max: " << initial_batch_size_max << endl;
    cout << "\tbatch_size_min: " << batch_size_min << endl;
    cout << "\tbatch_size_max: " << batch_size_max << endl;

    cout << "\tinitial_mu_min: " << initial_mu_min << endl;
    cout << "\tinitial_mu_max: " << initial_mu_max << endl;
    cout << "\tmu_min: " << mu_min << endl;
    cout << "\tmu_max: " << mu_max << endl;

    cout << "\tinitial_mu_delta_min: " << initial_mu_delta_min << endl;
    cout << "\tinitial_mu_delta_max: " << initial_mu_delta_max << endl;
    cout << "\tmu_delta_min: " << mu_delta_min << endl;
    cout << "\tmu_delta_max: " << mu_delta_max << endl;

    cout << "\tinitial_learning_rate_min: " << initial_learning_rate_min << endl;
    cout << "\tinitial_learning_rate_max: " << initial_learning_rate_max << endl;
    cout << "\tlearning_rate_min: " << learning_rate_min << endl;
    cout << "\tlearning_rate_max: " << learning_rate_max << endl;

    cout << "\tinitial_learning_rate_delta_min: " << initial_learning_rate_delta_min << endl;
    cout << "\tinitial_learning_rate_delta_max: " << initial_learning_rate_delta_max << endl;
    cout << "\tlearning_rate_delta_min: " << learning_rate_delta_min << endl;
    cout << "\tlearning_rate_delta_max: " << learning_rate_delta_max << endl;

    cout << "\tinitial_weight_decay_min: " << initial_weight_decay_min << endl;
    cout << "\tinitial_weight_decay_max: " << initial_weight_decay_max << endl;
    cout << "\tweight_decay_min: " << weight_decay_min << endl;
    cout << "\tweight_decay_max: " << weight_decay_max << endl;

    cout << "\tinitial_weight_decay_delta_min: " << initial_weight_decay_delta_min << endl;
    cout << "\tinitial_weight_decay_delta_max: " << initial_weight_decay_delta_max << endl;
    cout << "\tweight_decay_delta_min: " << weight_decay_delta_min << endl;
    cout << "\tweight_decay_delta_max: " << weight_decay_delta_max << endl;

    cout << "\tepsilon: " << epsilon << endl;

    cout << "\tinitial_alpha_min: " << initial_alpha_min << endl;
    cout << "\tinitial_alpha_max: " << initial_alpha_max << endl;
    cout << "\talpha_min: " << alpha_min << endl;
    cout << "\talpha_max: " << alpha_max << endl;

    cout << "\tinitial_velocity_reset_min: " << initial_velocity_reset_min << endl;
    cout << "\tinitial_velocity_reset_max: " << initial_velocity_reset_max << endl;
    cout << "\tvelocity_reset_min: " << velocity_reset_min << endl;
    cout << "\tvelocity_reset_max: " << velocity_reset_max << endl;

    cout << "\tinitial_input_dropout_probability_min: " << initial_input_dropout_probability_min << endl;
    cout << "\tinitial_input_dropout_probability_max: " << initial_input_dropout_probability_max << endl;
    cout << "\tinput_dropout_probability_min: " << input_dropout_probability_min << endl;
    cout << "\tinput_dropout_probability_max: " << input_dropout_probability_max << endl;

    cout << "\tinitial_hidden_dropout_probability_min: " << initial_hidden_dropout_probability_min << endl;
    cout << "\tinitial_hidden_dropout_probability_max: " << initial_hidden_dropout_probability_max << endl;
    cout << "\thidden_dropout_probability_min: " << hidden_dropout_probability_min << endl;
    cout << "\thidden_dropout_probability_max: " << hidden_dropout_probability_max << endl;

    cout << "\tmax_epochs: " << max_epochs << endl;
    cout << "\treset_weights_chance: " << reset_weights_chance << endl;

    cout << "\tcrossover_settings: " << endl;
    cout << "\t\tcrossover_rate: " << crossover_rate << endl;
    cout << "\t\tmore_fit_parent_crossover: " << more_fit_parent_crossover << endl;
    cout << "\t\tless_fit_parent_crossover: " << less_fit_parent_crossover << endl;
    cout << "\t\tcrossover_alter_edge_type: " << crossover_alter_edge_type << endl;

    cout << "\tmutation_settings: " << endl;
    cout << "\t\tnumber_mutations: " << number_mutations << endl;
    cout << "\t\tedge_alter_type: " << edge_alter_type << endl;
    cout << "\t\tedge_disable: " << edge_disable << endl;
    cout << "\t\tedge_split: " << edge_split << endl;
    cout << "\t\tedge_add: " << edge_add << endl;
    cout << "\t\tnode_change_size: " << node_change_size << endl;
    cout << "\t\tnode_change_size_x: " << node_change_size_x << endl;
    cout << "\t\tnode_change_size_y: " << node_change_size_y << endl;
    cout << "\t\tnode_add: " << node_add << endl;
    cout << "\t\tnode_split: " << node_split << endl;
    cout << "\t\tnode_merge: " << node_merge << endl;
    cout << "\t\tnode_enable: " << node_enable << endl;
    cout << "\t\tnode_disable: " << node_disable << endl;

    float total = edge_disable + edge_enable + edge_split + edge_add + edge_alter_type +
                   node_change_size + node_change_size_x + node_change_size_y + node_add +
                   node_split + node_merge + node_enable + node_disable;

    edge_alter_type /= total;
    edge_disable /= total;
    edge_enable /= total;
    edge_split /= total;
    edge_add /= total;
    node_change_size /= total;
    node_change_size_x /= total;
    node_change_size_y /= total;
    node_add /= total;
    node_split /= total;
    node_merge /= total;
    node_enable /= total;
    node_disable /= total;

    cout << "mutation probabilities: " << endl;
    cout << "\tedge_alter_type: " << edge_alter_type << endl;
    cout << "\tedge_disable: " << edge_disable << endl;
    cout << "\tedge_split: " << edge_split << endl;
    cout << "\tedge_add: " << edge_add << endl;
    cout << "\tnode_change_size: " << node_change_size << endl;
    cout << "\tnode_change_size_x: " << node_change_size_x << endl;
    cout << "\tnode_change_size_y: " << node_change_size_y << endl;
    cout << "\tnode_add: " << node_add << endl;
    cout << "\tnode_split: " << node_split << endl;
    cout << "\tnode_merge: " << node_merge << endl;
    cout << "\tnode_enable: " << node_enable << endl;
    cout << "\tnode_disable: " << node_disable << endl;

    if (output_directory.compare("") != 0) {
        write_statistics_header();
        write_hyperparameters_header();
    }
}

int EXACT::get_id() const {
    return id;
}

int EXACT::get_inserted_genomes() const {
    return inserted_genomes;
}

int EXACT::get_max_genomes() const {
    return max_genomes;
}


string EXACT::get_search_name() const {
    return search_name;
}

string EXACT::get_output_directory() const {
    return output_directory;
}

string EXACT::get_training_filename() const {
    return training_filename;
}

string EXACT::get_validation_filename() const {
    return validation_filename;
}

string EXACT::get_test_filename() const {
    return test_filename;
}



int EXACT::get_number_training_images() const {
    return number_training_images;
}

CNN_Genome* EXACT::get_best_genome() {
    return genomes[0];
}

int EXACT::get_number_genomes() const {
    return genomes.size();
}

CNN_Genome* EXACT::get_genome(int i) {
    return genomes[i];
}

void EXACT::generate_initial_hyperparameters(float &mu, float &mu_delta, float &learning_rate, float &learning_rate_delta, float &weight_decay, float &weight_decay_delta, float &alpha, int &velocity_reset, float &input_dropout_probability, float &hidden_dropout_probability, int &batch_size) {
    mu = (rng_float(generator) * (initial_mu_max - initial_mu_min)) + initial_mu_min;
    mu_delta = (rng_float(generator) * (initial_mu_delta_max - initial_mu_delta_min)) + initial_mu_delta_min;

    learning_rate = (rng_float(generator) * (initial_learning_rate_max - initial_learning_rate_min)) + initial_learning_rate_min;
    learning_rate_delta = (rng_float(generator) * (initial_learning_rate_delta_max - initial_learning_rate_delta_min)) + initial_learning_rate_delta_min;

    weight_decay = (rng_float(generator) * (initial_weight_decay_max - initial_weight_decay_min)) + initial_weight_decay_min;
    weight_decay_delta = (rng_float(generator) * (initial_weight_decay_delta_max - initial_weight_decay_delta_min)) + initial_weight_decay_delta_min;

    alpha = (rng_float(generator) * (initial_alpha_max - initial_alpha_min)) + initial_alpha_min;

    velocity_reset = (rng_float(generator) * (initial_velocity_reset_max - initial_velocity_reset_min)) + initial_velocity_reset_min;

    input_dropout_probability = (rng_float(generator) * (initial_input_dropout_probability_max - initial_input_dropout_probability_min)) + initial_input_dropout_probability_min;
    hidden_dropout_probability = (rng_float(generator) * (initial_hidden_dropout_probability_max - initial_hidden_dropout_probability_min)) + initial_hidden_dropout_probability_min;

    batch_size = (rng_float(generator) * (initial_batch_size_max - initial_batch_size_min)) + initial_batch_size_min;

    cout << "\tGenerated RANDOM hyperparameters:" << endl;
    cout << "\t\tmu: " << mu << endl;
    cout << "\t\tmu_delta: " << mu_delta << endl;
    cout << "\t\tlearning_rate: " << learning_rate << endl;
    cout << "\t\tlearning_rate_delta: " << learning_rate_delta << endl;
    cout << "\t\tweight_decay: " << weight_decay << endl;
    cout << "\t\tweight_decay_delta: " << weight_decay_delta << endl;
    cout << "\t\talpha: " << alpha << endl;
    cout << "\t\tvelocity_reset: " << velocity_reset << endl;
    cout << "\t\tinput_dropout_probability: " << input_dropout_probability << endl;
    cout << "\t\thidden_dropout_probability: " << hidden_dropout_probability << endl;
    cout << "\t\tbatch_size: " << batch_size << endl;
}

void EXACT::generate_simplex_hyperparameters(float &mu, float &mu_delta, float &learning_rate, float &learning_rate_delta, float &weight_decay, float &weight_decay_delta, float &alpha, int &velocity_reset, float &input_dropout_probability, float &hidden_dropout_probability, int &batch_size) {

    float best_mu, best_mu_delta, best_learning_rate, best_learning_rate_delta, best_weight_decay, best_weight_decay_delta, best_alpha, best_velocity_reset, best_input_dropout_probability, best_hidden_dropout_probability, best_batch_size;

    //get best hyperparameters
    //now getting best of group instead of overall best
    /*
    CNN_Genome *best_genome = genomes[0];
    best_mu = best_genome->get_initial_mu();
    best_mu_delta = best_genome->get_mu_delta();
    best_learning_rate = best_genome->get_initial_learning_rate();
    best_learning_rate_delta = best_genome->get_learning_rate_delta();
    best_weight_decay = best_genome->get_initial_weight_decay();
    best_weight_decay_delta = best_genome->get_weight_decay_delta();
    best_alpha = best_genome->get_alpha();
    best_velocity_reset = best_genome->get_velocity_reset();
    best_input_dropout_probability = best_genome->get_input_dropout_probability();
    best_hidden_dropout_probability = best_genome->get_hidden_dropout_probability();
    best_batch_size = best_genome->get_batch_size();
    */

    //get average parameters
    float avg_mu, avg_mu_delta, avg_learning_rate, avg_learning_rate_delta, avg_weight_decay, avg_weight_decay_delta, avg_alpha, avg_velocity_reset, avg_input_dropout_probability, avg_hidden_dropout_probability, avg_batch_size;

    avg_mu = 0;
    avg_mu_delta = 0;
    avg_learning_rate = 0;
    avg_learning_rate_delta = 0;
    avg_weight_decay = 0;
    avg_weight_decay_delta = 0;
    avg_learning_rate = 0;
    avg_alpha = 0;
    avg_velocity_reset = 0;
    avg_input_dropout_probability = 0;
    avg_hidden_dropout_probability = 0;
    avg_batch_size = 0;

    float best_fitness = EXACT_MAX_FLOAT;
    int simplex_count = 5;
    for (uint32_t i = 0; i < simplex_count; i++) {
        CNN_Genome *current_genome = genomes[rng_float(generator) * genomes.size()];

        //getting best parameters from group instead of best of population
        if (i == 0 || current_genome->get_best_validation_error() < best_fitness) {
            best_mu = current_genome->get_initial_mu();
            best_mu_delta = current_genome->get_mu_delta();
            best_learning_rate = current_genome->get_initial_learning_rate();
            best_learning_rate_delta = current_genome->get_learning_rate_delta();
            best_weight_decay = current_genome->get_initial_weight_decay();
            best_weight_decay_delta = current_genome->get_weight_decay_delta();
            best_alpha = current_genome->get_alpha();
            best_velocity_reset = current_genome->get_velocity_reset();
            best_input_dropout_probability = current_genome->get_input_dropout_probability();
            best_hidden_dropout_probability = current_genome->get_hidden_dropout_probability();
            best_batch_size = current_genome->get_batch_size();
        }

        avg_mu += current_genome->get_initial_mu();
        avg_mu_delta += current_genome->get_mu_delta();
        avg_learning_rate += current_genome->get_initial_learning_rate();
        avg_learning_rate_delta += current_genome->get_learning_rate_delta();
        avg_weight_decay += current_genome->get_initial_weight_decay();
        avg_weight_decay_delta += current_genome->get_weight_decay_delta();
        avg_alpha += current_genome->get_alpha();
        avg_velocity_reset += current_genome->get_velocity_reset();
        avg_input_dropout_probability += current_genome->get_input_dropout_probability();
        avg_hidden_dropout_probability += current_genome->get_hidden_dropout_probability();
        avg_batch_size += current_genome->get_batch_size();
    }

    avg_mu /= simplex_count;
    avg_mu_delta /= simplex_count;
    avg_learning_rate /= simplex_count;
    avg_learning_rate_delta /= simplex_count;
    avg_weight_decay /= simplex_count;
    avg_weight_decay_delta /= simplex_count;
    avg_learning_rate /= simplex_count;
    avg_alpha /= simplex_count;
    avg_velocity_reset /= simplex_count;
    avg_input_dropout_probability /= simplex_count;
    avg_hidden_dropout_probability /= simplex_count;
    avg_batch_size /= simplex_count;

    float scale = (rng_float(generator) * 2.0) - 0.5;

    mu = avg_mu + ((best_mu - avg_mu) * scale);
    mu_delta = avg_mu_delta + ((best_mu_delta - avg_mu_delta) * scale);
    learning_rate = avg_learning_rate + ((best_learning_rate - avg_learning_rate) * scale);
    learning_rate_delta = avg_learning_rate_delta + ((best_learning_rate_delta - avg_learning_rate_delta) * scale);
    weight_decay = avg_weight_decay + ((best_weight_decay - avg_weight_decay) * scale);
    weight_decay_delta = avg_weight_decay_delta + ((best_weight_decay_delta - avg_weight_decay_delta) * scale);
    learning_rate = avg_learning_rate + ((best_learning_rate - avg_learning_rate) * scale);
    alpha = avg_alpha + ((best_alpha - avg_alpha) * scale);
    velocity_reset = avg_velocity_reset + ((best_velocity_reset - avg_velocity_reset) * scale);
    input_dropout_probability = avg_input_dropout_probability + ((best_input_dropout_probability - avg_input_dropout_probability) * scale);
    hidden_dropout_probability = avg_hidden_dropout_probability + ((best_hidden_dropout_probability - avg_hidden_dropout_probability) * scale);
    batch_size = avg_batch_size + ((best_batch_size - avg_batch_size) * scale);

    if (mu < mu_min) mu = mu_min;
    if (mu > mu_max) mu = mu_max;
    if (mu_delta < mu_delta_min) mu_delta = mu_delta_min;
    if (mu_delta > mu_delta_max) mu_delta = mu_delta_max;

    if (learning_rate < learning_rate_min) learning_rate = learning_rate_min;
    if (learning_rate > learning_rate_max) learning_rate = learning_rate_max;
    if (learning_rate_delta < learning_rate_delta_min) learning_rate_delta = learning_rate_delta_min;
    if (learning_rate_delta > learning_rate_delta_max) learning_rate_delta = learning_rate_delta_max;

    if (weight_decay < weight_decay_min) weight_decay = weight_decay_min;
    if (weight_decay > weight_decay_max) weight_decay = weight_decay_max;
    if (weight_decay_delta < weight_decay_delta_min) weight_decay_delta = weight_decay_delta_min;
    if (weight_decay_delta > weight_decay_delta_max) weight_decay_delta = weight_decay_delta_max;

    if (alpha < alpha_min) alpha = alpha_min;
    if (alpha > alpha_max) alpha = alpha_max;

    if (velocity_reset < velocity_reset_min) velocity_reset = velocity_reset_min;
    if (velocity_reset > velocity_reset_max) velocity_reset = velocity_reset_max;

    if (input_dropout_probability < input_dropout_probability_min) input_dropout_probability = input_dropout_probability_min;
    if (input_dropout_probability > input_dropout_probability_max) input_dropout_probability = input_dropout_probability_max;
    if (hidden_dropout_probability < hidden_dropout_probability_min) hidden_dropout_probability = hidden_dropout_probability_min;
    if (hidden_dropout_probability > hidden_dropout_probability_max) hidden_dropout_probability = hidden_dropout_probability_max;

    if (batch_size < batch_size_min) batch_size = batch_size_min;
    if (batch_size > batch_size_max) batch_size = batch_size_max;

    cout << "\tGenerated SIMPLEX hyperparameters:" << endl;
    cout << "\t\tscale: " << scale << endl;
    cout << "\t\tmu: " << mu << endl;
    cout << "\t\tmu_delta: " << mu_delta << endl;
    cout << "\t\tlearning_rate: " << learning_rate << endl;
    cout << "\t\tlearning_rate_delta: " << learning_rate_delta << endl;
    cout << "\t\tweight_decay: " << weight_decay << endl;
    cout << "\t\tweight_decay_delta: " << weight_decay_delta << endl;
    cout << "\t\talpha: " << alpha << endl;
    cout << "\t\tvelocity_reset: " << velocity_reset << endl;
    cout << "\t\tinput_dropout_probability: " << input_dropout_probability << endl;
    cout << "\t\thidden_dropout_probability: " << hidden_dropout_probability << endl;
    cout << "\t\tbatch_size: " << batch_size << endl;
}



CNN_Genome* EXACT::generate_individual() {
    if (inserted_genomes >= max_genomes) return NULL;

    CNN_Genome *genome = NULL;
    if (genomes.size() == 0) {
        //generate initial random hyperparameters
        float mu, mu_delta, learning_rate, learning_rate_delta, weight_decay, weight_decay_delta, alpha, input_dropout_probability, hidden_dropout_probability;
        int velocity_reset, batch_size;

        generate_initial_hyperparameters(mu, mu_delta, learning_rate, learning_rate_delta, weight_decay, weight_decay_delta, alpha, velocity_reset, input_dropout_probability, hidden_dropout_probability, batch_size);

        //generate the initial minimal CNN
        vector<CNN_Node*> genome_nodes;

        vector<CNN_Node*> input_nodes;
        for (int32_t i = 0; i < image_channels; i++) {
            CNN_Node *input_node = new CNN_Node(node_innovation_count, 0, batch_size, image_rows, image_cols, INPUT_NODE);
            node_innovation_count++;
            genome_nodes.push_back(input_node);
            input_nodes.push_back(input_node);
        }

        for (int32_t i = 0; i < number_classes; i++) {
            CNN_Node *softmax_node = new CNN_Node(node_innovation_count, 1, batch_size, 1, 1, SOFTMAX_NODE);
            node_innovation_count++;
            genome_nodes.push_back(softmax_node);
        }

        vector<CNN_Edge*> genome_edges;
        for (int32_t i = 0; i < number_classes; i++) {
            for (int32_t j = 0; j < image_channels; j++) {
                CNN_Edge *edge = new CNN_Edge(input_nodes[j], genome_nodes[i + image_channels] /*ith softmax node*/, true, edge_innovation_count, CONVOLUTIONAL);

                genome_edges.push_back(edge);

                edge_innovation_count++;
            }
        }

        long genome_seed = rng_long(generator);
        //cout << "seeding genome with: " << genome_seed << endl;

        genome = new CNN_Genome(genomes_generated++, padding, number_training_images, number_validation_images, number_test_images, genome_seed, max_epochs, reset_weights, velocity_reset, mu, mu_delta, learning_rate, learning_rate_delta, weight_decay, weight_decay_delta, batch_size, epsilon, alpha, input_dropout_probability, hidden_dropout_probability, genome_nodes, genome_edges);

    } else if ((int32_t)genomes.size() < population_size) {
        //generate random mutatinos until genomes.size() < population_size
        while (genome == NULL) {
            genome = create_mutation();

            if (!genome->visit_nodes()) {
                cout << "\tAll softmax nodes were not reachable, deleting genome." << endl;
                delete genome;
                genome = NULL;
            }
        }
    } else {
        if (rng_float(generator) < crossover_rate) {
            //generate a child from crossover
            while (genome == NULL) {
                genome = create_child();

                if (!genome->visit_nodes()) {
                    cout << "\tAll softmax nodes were not reachable, deleting genome." << endl;
                    delete genome;
                    genome = NULL;
                }
            }

        } else {
            //generate a mutation
            while (genome == NULL) {
                genome = create_mutation();

                if (!genome->visit_nodes()) {
                    cout << "\tAll softmax nodes were not reachable, deleting genome." << endl;
                    delete genome;
                    genome = NULL;
                }
            }
        }
    }

    genome->initialize();

    if (!genome->sanity_check(SANITY_CHECK_AFTER_GENERATION)) {
        cout << "ERROR: genome " << genome->get_generation_id() << " failed sanity check in generate individual!" << endl;
        exit(1);
    }

    if ((int32_t)genomes.size() < population_size) {
        //insert a copy with a bad fitness so we have more things to generate new genomes with
        vector<CNN_Node*> node_copies;
        vector<CNN_Edge*> edge_copies;
        genome->get_node_copies(node_copies);
        genome->get_edge_copies(edge_copies);

        cout << "creating genome_copy to insert into population because genomes.size() < population_size" << endl;
        CNN_Genome *genome_copy = new CNN_Genome(genomes_generated++, padding, number_training_images, number_validation_images, number_test_images, /*new random seed*/ rng_long(generator), max_epochs, reset_weights, genome->get_velocity_reset(), genome->get_initial_mu(), genome->get_mu_delta(), genome->get_initial_learning_rate(), genome->get_learning_rate_delta(), genome->get_initial_weight_decay(), genome->get_weight_decay_delta(), genome->get_batch_size(), epsilon, genome->get_alpha(), genome->get_input_dropout_probability(), genome->get_hidden_dropout_probability(), node_copies, edge_copies);
        genome_copy->initialize();

        //for more variability in the initial population, re-initialize weights and bias for these unevaluated copies

        insert_genome(genome_copy);
    }

    return genome;
}

int32_t EXACT::population_contains(CNN_Genome *genome) const {
    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        //we can overwrite genomes that were inserted in the initialization phase
        //and not evaluated
        //if (genomes[i]->get_best_validation_error() == EXACT_MAX_FLOAT) continue;

        if (genomes[i]->equals(genome)) {
            cout << "\tgenome was the same as genome with generation id: " << genomes[i]->get_generation_id() << endl;
            return i;
        }
    }

    return -1;
}

string parse_fitness(float fitness) {
    if (fitness == EXACT_MAX_FLOAT) {
        return "UNEVALUATED";
    } else {
        return to_string(fitness);
    }
}

bool EXACT::insert_genome(CNN_Genome* genome) {
    float new_fitness = genome->get_best_validation_error();
    int new_generation_id = genome->get_generation_id();

    bool was_inserted = true;

    bool was_best_predictions_genome = false;

    inserted_genomes++;

    if (genome->get_best_validation_error() != EXACT_MAX_FLOAT) {
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
        gv_file << "#EXACT settings: " << endl;

        gv_file << "#EXACT settings: " << endl;

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

bool EXACT::add_edge(CNN_Genome *child, CNN_Node *node1, CNN_Node *node2, int edge_type) {
    int node1_innovation_number = node1->get_innovation_number();
    int node2_innovation_number = node2->get_innovation_number();

    //check to see if the edge already exists
    for (int32_t i = 0; i < child->get_number_edges(); i++) {
        CNN_Edge *edge = child->get_edge(i);
        if (edge->connects(node1_innovation_number, node2_innovation_number)) {

            if (edge->is_disabled()) {
                edge->enable();
                edge->set_needs_init();
                if (!edge->set_nodes(child->get_nodes())) {
                    edge->resize();
                }

                return true;
            } else {
                return false;
            }
        }
    }

    //edge doesn't exist, add it
    cout << "\t\tadding edge between node innovation numbers " << node1_innovation_number << " and " << node2_innovation_number << endl;

    CNN_Edge *edge = NULL;

    if (use_sfmp) {
        edge = new CNN_Edge(node1, node2, false, edge_innovation_count, edge_type);
    } else {
        edge = new CNN_Edge(node1, node2, false, edge_innovation_count, CONVOLUTIONAL);
    }
    edge_innovation_count++;

    //insert edge in order of depth
    child->add_edge(edge);

    return true;
}


CNN_Genome* EXACT::create_mutation() {
    //mutation options:
    //edges:
    //  1. disable edge (but make sure output node is still reachable)
    //  2. split edge
    //  3. add edge (make sure it does not exist already)
    //  4. increase/decrease stride (not yet)
    //nodes:
    //  1. increase/decrease size_x
    //  2. increase/decrease size_y
    //  3. increase/decrease max_pool (not yet)

    long child_seed = rng_long(generator);

    CNN_Genome *parent = genomes[rng_float(generator) * genomes.size()];

    cout << "\tgenerating child " << genomes_generated << " from parent genome: " << parent->get_generation_id() << endl;

    float mu, mu_delta, learning_rate, learning_rate_delta, weight_decay, weight_decay_delta, alpha, input_dropout_probability, hidden_dropout_probability;
    int velocity_reset, batch_size;

    if (inserted_genomes < (population_size * 10)) {
        cout << "\tGenerating hyperparameters randomly." << endl;
        generate_initial_hyperparameters(mu, mu_delta, learning_rate, learning_rate_delta, weight_decay, weight_decay_delta, alpha, velocity_reset, input_dropout_probability, hidden_dropout_probability, batch_size);
    } else {
        cout << "\tGenerating hyperparameters with simplex." << endl;
        generate_simplex_hyperparameters(mu, mu_delta, learning_rate, learning_rate_delta, weight_decay, weight_decay_delta, alpha, velocity_reset, input_dropout_probability, hidden_dropout_probability, batch_size);
    }

    vector<CNN_Node*> node_copies;
    vector<CNN_Edge*> edge_copies;
    parent->get_node_copies(node_copies);
    parent->get_edge_copies(edge_copies);

    CNN_Genome *child = new CNN_Genome(genomes_generated++, padding, number_training_images, number_validation_images, number_test_images, child_seed, max_epochs, reset_weights, velocity_reset, mu, mu_delta, learning_rate, learning_rate_delta, weight_decay, weight_decay_delta, batch_size, epsilon, alpha, input_dropout_probability, hidden_dropout_probability, node_copies, edge_copies);

    /*
    cout << "\tchild nodes:" << endl;
    for (int32_t i = 0; i < child->get_number_nodes(); i++) {
        cout << "\t\tnode innovation number: " << child->get_node(i)->get_innovation_number() << endl;
    }

    cout << "\tchild edges:" << endl;
    for (int32_t i = 0; i < child->get_number_edges(); i++) {
        cout << "\t\tedge innovation number: " << child->get_edge(i)->get_innovation_number()
            << ", input node innovation number: " << child->get_edge(i)->get_input_innovation_number()
            << ", output node innovation number: " << child->get_edge(i)->get_output_innovation_number()
            << endl;
    }
    */

    if (parent->get_best_validation_error() == EXACT_MAX_FLOAT) {
        //This parent has not actually been evaluated (the population is still initializing)
        //we can set the best_bias and best_weights randomly so that they are used when it
        //starts up

        cout << "\tparent had not been evaluated yet, but best_bias and best_weights should have been set randomly" << endl;
    } else {
        cout << "\tparent had been evaluated! not setting best_bias and best_weights randomly" << endl;
        cout << "\tparent fitness: " << parent->get_best_validation_error() << endl;
    }

    int modifications = 0;

    while (modifications < number_mutations) {
        child->visit_nodes();

        float r = rng_float(generator);
        cout << "\tr: " << r << endl;
        r -= 0.00001;

        if (r < edge_alter_type) {
            cout << "\tALTERING EDGE TYPE!" << endl;
            vector< CNN_Edge* > reachable_edges = child->get_reachable_edges();

            if (reachable_edges.size() == 0) {
                cout << "\t\tcould not alter edge type as there were no enabled edges!" << endl;
                cout << "\t\tthis should never happen!" << endl;
                exit(1);
            }

            int edge_position = rng_float(generator) * reachable_edges.size();
            CNN_Edge* reachable_edge = reachable_edges[edge_position];

            cout << "\t\taltering edge type on edge: " << reachable_edge->get_innovation_number() << " between input node innovation number " << reachable_edge->get_input_node()->get_innovation_number() << " and output node innovation number " << reachable_edge->get_output_node()->get_innovation_number() << endl;

            reachable_edge->alter_edge_type();
            //reinitialize weights for re-enabled edge
            reachable_edge->set_needs_init();
            child->set_generated_by("alter_edge_type");
            modifications++;
            
            continue;
        }
        r -= edge_alter_type;

        if (r < edge_disable) {
            cout << "\tDISABLING EDGE!" << endl;

            vector< CNN_Edge* > reachable_edges = child->get_reachable_edges();

            if (reachable_edges.size() == 0) {
                cout << "\t\tno reachable edges! this should never happen!" << endl;
                exit(1);
                continue;
            }

            int edge_position = rng_float(generator) * reachable_edges.size();

            reachable_edges[edge_position]->disable();
            child->set_generated_by("disable_edge");
            modifications++;

            continue;
        } 
        r -= edge_disable;

        if (r < edge_enable) {
            cout << "\tENABLING EDGE!" << endl;

            vector< CNN_Edge* > disabled_edges = child->get_disabled_edges();

            if (disabled_edges.size() == 0) {
                cout << "\t\tcould not enable an edge as there were no disabled edges!" << endl;
                continue;
            }

            int edge_position = rng_float(generator) * disabled_edges.size();

            CNN_Edge* disabled_edge = disabled_edges[edge_position];

            cout << "\t\tenabling edge: " << disabled_edge->get_innovation_number() << " between input node innovation number " << disabled_edge->get_input_node()->get_innovation_number() << " and output node innovation number " << disabled_edge->get_output_node()->get_innovation_number() << endl;

            disabled_edge->enable();
            //reinitialize weights for re-enabled edge
            disabled_edge->set_needs_init();
            child->set_generated_by("enable_edge");
            modifications++;

            continue;
        } 
        r -= edge_enable;


        if (r < edge_split) {
            vector< CNN_Edge* > reachable_edges = child->get_reachable_edges();

            if (reachable_edges.size() == 0) {
                cout << "\t\tno reachable edges! this should never happen!" << endl;
                exit(1);
            }

            int edge_position = rng_float(generator) * reachable_edges.size();

            CNN_Edge* edge = reachable_edges[edge_position];

            CNN_Node* input_node = edge->get_input_node();
            CNN_Node* output_node = edge->get_output_node();

            float depth = (input_node->get_depth() + output_node->get_depth()) / 2.0;
            int size_x = (input_node->get_size_x() + output_node->get_size_x()) / 2.0;
            int size_y = (input_node->get_size_y() + output_node->get_size_y()) / 2.0;

            CNN_Node *child_node = new CNN_Node(node_innovation_count, depth, batch_size, size_x, size_y, HIDDEN_NODE);
            node_innovation_count++;

            //add two new edges, disable the split edge
            CNN_Edge *edge1 = NULL;
            cout << "\t\tcreating edge " << edge_innovation_count << endl;
            if (use_sfmp) {
                edge1 = new CNN_Edge(input_node, child_node, false, edge_innovation_count, random_edge_type(rng_float(generator)));
            } else {
                edge1 = new CNN_Edge(input_node, child_node, false, edge_innovation_count, CONVOLUTIONAL);
            }
            edge_innovation_count++;

            CNN_Edge *edge2 = NULL;
            cout << "\t\tcreating edge " << edge_innovation_count << endl;
            if (use_sfmp) {
                edge2 = new CNN_Edge(child_node, output_node, false, edge_innovation_count, random_edge_type(rng_float(generator)));
            } else {
                edge2 = new CNN_Edge(child_node, output_node, false, edge_innovation_count, CONVOLUTIONAL);
            }
            edge_innovation_count++;

            cout << "\t\tdisabling edge " << edge->get_innovation_number() << endl;
            edge->disable();

            child->add_node(child_node);
            child->add_edge(edge1);
            child->add_edge(edge2);

            child->set_generated_by("split_edge");
            modifications++;

            continue;
        }
        r -= edge_split;

        if (r < edge_add) {
            cout << "\tADDING EDGE!" << endl;

            vector< CNN_Node* > reachable_nodes = child->get_reachable_nodes();

            if (reachable_nodes.size() < 2) {
                cout << "\t\tless than 2 reachable nodes! this should never happen!" << endl;
                exit(1);
                continue;
            }

            CNN_Node *node1;
            CNN_Node *node2;

            do {
                int r1 = rng_float(generator) * reachable_nodes.size();
                int r2 = rng_float(generator) * reachable_nodes.size() - 1;

                if (r1 == r2) r2++;

                if (r1 > r2) {  //swap r1 and r2 so node2 is always deeper than node1
                    int temp = r1;
                    r1 = r2;
                    r2 = temp;
                }

                //cout << "child->get_number_nodes(): " <<  child->get_number_nodes() << ", r1: " << r1 << ", r2: " << r2 << endl;

                node1 = reachable_nodes[r1];
                node2 = reachable_nodes[r2];
            } while (node1->get_depth() >= node2->get_depth());
            //after this while loop, node 2 will always be deeper than node 1

            if (use_sfmp) {
                if (add_edge(child, node1, node2, random_edge_type(rng_float(generator)))) {
                    child->set_generated_by("add_edge");
                    modifications++;
                } else {
                    cout << "\t\tnot adding edge between node innovation numbers " << node1->get_innovation_number() << " and " << node2->get_innovation_number() << " because edge already exists!" << endl;
                }
            } else {
                if (add_edge(child, node1, node2, CONVOLUTIONAL)) {
                    child->set_generated_by("add_edge");
                    modifications++;
                } else {
                    cout << "\t\tnot adding edge between node innovation numbers " << node1->get_innovation_number() << " and " << node2->get_innovation_number() << " because edge already exists!" << endl;
                }
            }

            continue;
        }
        r -= edge_add;

        if (r < node_change_size) {
            cout << "\tCHANGING NODE SIZE X and Y!" << endl;

            if (child->get_number_softmax_nodes() + child->get_number_input_nodes() == child->get_number_nodes()) {
                cout << "\t\tno non-input or softmax nodes so cannot change node size" << endl;
                continue;
            }

            //should have a value between -2 and 2 (inclusive)
            int change = (2 * rng_float(generator)) + 1;
            if (rng_float(generator) < 0.5) change *= -1;

            //make sure we don't change the size of the input node
            cout << "\t\tnumber nodes: " << child->get_number_nodes() << endl;
            cout << "\t\tnumber input nodes: " << child->get_number_input_nodes() << endl;
            cout << "\t\tnumber softmax nodes: " << child->get_number_softmax_nodes() << endl;

            vector<CNN_Node*> reachable_nodes = child->get_reachable_hidden_nodes();

            if (reachable_nodes.size() == 0) {
                cout << "\tthere were no reachable non-hidden nodes, cannot change a node size." << endl;
                continue;
            }
            
            cout << "\t\tthere are " << reachable_nodes.size() << " reachable nodes." << endl;

            int r = rng_float(generator) * reachable_nodes.size();
            cout << "\t\tr: " << r << endl;

            CNN_Node *modified_node = reachable_nodes[r];
            cout << "\t\tselected node: " << r << " with innovation number: " << modified_node->get_innovation_number() << endl;

            if (modified_node->is_input()) {
                cout << "\t\tmodified node was input, this should never happen!" << endl;
                exit(1);
            }

            if (modified_node->is_softmax()) {
                cout << "\t\tmodified node was softmax, this should never happen!" << endl;
                exit(1);
            }

            int previous_size_x = modified_node->get_size_x();
            int previous_size_y = modified_node->get_size_y();
            cout << "\t\tsize x before resize: " << previous_size_x << " modifying by change: " << change << endl;
            cout << "\t\tsize y before resize: " << previous_size_y << " modifying by change: " << change << endl;

            bool modified_x = modified_node->modify_size_x(change);
            bool modified_y = modified_node->modify_size_y(change);

            if (modified_x || modified_y) {
                //need to make sure all edges with this as it's input or output get updated
                cout << "\t\tresizing edges around node: " << modified_node->get_innovation_number() << endl;

                child->resize_edges_around_node( modified_node->get_innovation_number() );
                child->set_generated_by("change_size");
                modifications++;

                cout << "\t\tmodified size x by " << change << " from " << previous_size_x << " to " << modified_node->get_size_x() << endl;
                cout << "\t\tmodified size y by " << change << " from " << previous_size_y << " to " << modified_node->get_size_y() << endl;
            } else {
                cout << "\t\tmodification resulted in no change" << endl;
            }

            continue;
        }
        r -= node_change_size;

        if (r < node_change_size_x) {
            cout << "\tCHANGING NODE SIZE X!" << endl;

            if (child->get_number_softmax_nodes() + child->get_number_input_nodes() == child->get_number_nodes()) {
                cout << "\t\tno non-input or softmax nodes so cannot change node size" << endl;
                continue;
            }

            //should have a value between -2 and 2 (inclusive)
            int change = (2 * rng_float(generator)) + 1;
            if (rng_float(generator) < 0.5) change *= -1;

            //make sure we don't change the size of the input node
            cout << "\t\tnumber nodes: " << child->get_number_nodes() << endl;
            cout << "\t\tnumber input nodes: " << child->get_number_input_nodes() << endl;
            cout << "\t\tnumber softmax nodes: " << child->get_number_softmax_nodes() << endl;

            vector<CNN_Node*> reachable_nodes = child->get_reachable_hidden_nodes();

            if (reachable_nodes.size() == 0) {
                cout << "\tthere were no reachable non-hidden nodes, cannot change a node size." << endl;
                continue;
            }

            cout << "\t\tthere are " << reachable_nodes.size() << " reachable nodes." << endl;

            int r = rng_float(generator) * reachable_nodes.size();
            cout << "\t\tr: " << r << endl;

            CNN_Node *modified_node = reachable_nodes[r];
            cout << "\t\tselected node: " << r << " with innovation number: " << modified_node->get_innovation_number() << endl;

            if (modified_node->is_input()) {
                cout << "\t\tmodified node was input, this should never happen!" << endl;
                exit(1);
            }

            if (modified_node->is_softmax()) {
                cout << "\t\tmodified node was softmax, this should never happen!" << endl;
                exit(1);
            }


            int previous_size_x = modified_node->get_size_x();
            cout << "\t\tsize x before resize: " << previous_size_x << " modifying by change: " << change << endl;

            if (modified_node->modify_size_x(change)) {
                //need to make sure all edges with this as it's input or output get updated
                cout << "\t\tresizing edges around node: " << modified_node->get_innovation_number() << endl;

                child->resize_edges_around_node( modified_node->get_innovation_number() );
                child->set_generated_by("change_size_x");
                modifications++;

                cout << "\t\tmodified size x by " << change << " from " << previous_size_x << " to " << modified_node->get_size_x() << endl;
            } else {
                cout << "\t\tmodification resulted in no change" << endl;
            }

            continue;
        }
        r -= node_change_size_x;

        if (r < node_change_size_y) {
            cout << "\tCHANGING NODE SIZE Y!" << endl;

            if (child->get_number_softmax_nodes() + child->get_number_input_nodes() == child->get_number_nodes()) {
                cout << "\t\tno non-input or softmax nodes so cannot change node size" << endl;
                continue;
            }

            //should have a value between -2 and 2 (inclusive)
            int change = (2 * rng_float(generator)) + 1;
            if (rng_float(generator) < 0.5) change *= -1;

            //make sure we don't change the size of the input node
            cout << "\t\tnumber nodes: " << child->get_number_nodes() << endl;
            cout << "\t\tnumber input nodes: " << child->get_number_input_nodes() << endl;
            cout << "\t\tnumber softmax nodes: " << child->get_number_softmax_nodes() << endl;

            vector<CNN_Node*> reachable_nodes = child->get_reachable_hidden_nodes();

            if (reachable_nodes.size() == 0) {
                cout << "\tthere were no reachable non-hidden nodes, cannot change a node size." << endl;
                continue;
            }

            cout << "\t\tthere are " << reachable_nodes.size() << " reachable nodes." << endl;

            int r = rng_float(generator) * reachable_nodes.size();
            cout << "\t\tr: " << r << endl;

            CNN_Node *modified_node = reachable_nodes[r];
            cout << "\t\tselected node: " << r << " with innovation number: " << modified_node->get_innovation_number() << endl;

            if (modified_node->is_input()) {
                cout << "\t\tmodified node was input, this should never happen!" << endl;
                exit(1);
            }

            if (modified_node->is_softmax()) {
                cout << "\t\tmodified node was softmax, this should never happen!" << endl;
                exit(1);
            }


            int previous_size_y = modified_node->get_size_y();
            cout << "\t\tsize y before resize: " << previous_size_y << " modifying by change: " << change << endl;

            if (modified_node->modify_size_y(change)) {
                //need to make sure all edges with this as it's input or output get updated
                cout << "\t\tresizing edges around node: " << modified_node->get_innovation_number() << endl;

                child->resize_edges_around_node( modified_node->get_innovation_number() );
                child->set_generated_by("change_size_y");
                modifications++;

                cout << "\t\tmodified size y by " << change << " from " << previous_size_y << " to " << modified_node->get_size_y() << endl;
            } else {
                cout << "\t\tmodification resulted in no change" << endl;
            }

            continue;
        }
        r -= node_change_size_y;

        if (r < node_add) {
            cout << "\tADDING A NODE!" << endl;

            //pick random depth
            //separate nodes between those of less depth and those of greater depth

            float random_depth = rng_float(generator);
            cout << "\t\trandom depth: " << random_depth << endl;

            vector<CNN_Node*> potential_inputs;
            vector<CNN_Node*> potential_outputs;

            for (uint32_t i = 0; i < child->get_number_nodes(); i++) {

                if (child->get_node(i)->get_depth() < random_depth) {
                    potential_inputs.push_back(child->get_node(i));
                    cout << "\t\tnode " << i << " has depth: " << child->get_node(i)->get_depth() << " added as potential input" << endl;
                } else if (child->get_node(i)->get_depth() > random_depth) {
                    potential_outputs.push_back(child->get_node(i));
                    cout << "\t\tnode " << i << " has depth: " << child->get_node(i)->get_depth() << " added as potential output" << endl;
                } else {
                    cout << "\t\tnode " << i << " has depth: " << child->get_node(i)->get_depth() << " not added!" << endl;
                }
            }

            int number_inputs = (rng_float(generator) * 4) + 1;
            cout << "\t\tnumber inputs to use: " << number_inputs << endl;
            while (potential_inputs.size() > number_inputs) {
                potential_inputs.erase(potential_inputs.begin() + (rng_float(generator) * potential_inputs.size()));
            }

            if (potential_inputs.size() == 0) {
                cout << "\t\tNot adding node because no input nodes were selected." << endl;
                continue;
            }

            int number_outputs = (rng_float(generator) * 4) + 1;
            cout << "\t\tnumber outputs to use: " << number_outputs << endl;
            while (potential_outputs.size() > number_outputs) {
                potential_outputs.erase(potential_outputs.begin() + (rng_float(generator) * potential_outputs.size()));
            }

            if (potential_outputs.size() == 0) {
                cout << "\t\tNot adding node because no output nodes were selected." << endl;
                continue;
            }

            int32_t min_input_size_x = 1000, min_input_size_y = 1000, max_output_size_x = 0, max_output_size_y = 0;

            for (uint32_t i = 0; i < potential_inputs.size(); i++) {
                if (potential_inputs[i]->get_size_x() < min_input_size_x) min_input_size_x = potential_inputs[i]->get_size_x();
                if (potential_inputs[i]->get_size_y() < min_input_size_y) min_input_size_y = potential_inputs[i]->get_size_y();

                cout << "\t\tinput node: " << potential_inputs[i]->get_innovation_number() << endl;
            }

            for (uint32_t i = 0; i < potential_outputs.size(); i++) {
                if (potential_outputs[i]->get_size_x() > max_output_size_x) max_output_size_x = potential_outputs[i]->get_size_x();
                if (potential_outputs[i]->get_size_y() > max_output_size_y) max_output_size_y = potential_outputs[i]->get_size_y();
                cout << "\t\toutput node: " << potential_outputs[i]->get_innovation_number() << endl;
            }

            int32_t size_x = ((float)min_input_size_x + (float)max_output_size_x) / 2.0;
            int32_t size_y = ((float)min_input_size_y + (float)max_output_size_y) / 2.0;

            cout << "\t\tMin input size_x: " << min_input_size_x << ", size_y: " << min_input_size_y << endl;
            cout << "\t\tMax output size_x: " << max_output_size_x << ", size_y: " << max_output_size_y << endl;
            cout << "\t\tNew node will have size_x: " << size_x << ", size_y: " << size_y << endl;

            CNN_Node *child_node = new CNN_Node(node_innovation_count, random_depth, batch_size, size_x, size_y, HIDDEN_NODE);
            node_innovation_count++;

            child->add_node(child_node);

            int input_edge_type;
            int output_edge_type;

            if (use_sfmp) {
                input_edge_type = random_edge_type(rng_float(generator));
                output_edge_type = random_edge_type(rng_float(generator));
            } else {
                input_edge_type = CONVOLUTIONAL;
                output_edge_type = CONVOLUTIONAL;
            }

            for (uint32_t i = 0; i < potential_inputs.size(); i++) {
                add_edge(child, potential_inputs[i], child_node, input_edge_type);
            }

            for (uint32_t i = 0; i < potential_outputs.size(); i++) {
                add_edge(child, child_node, potential_outputs[i], output_edge_type);
            }

            child->set_generated_by("add_node");
            modifications++;

            continue;
        }
        r -= node_add;

        if (r < node_split) {
            cout << "\tSPLITTING A NODE!" << endl;

            vector<CNN_Node*> reachable_nodes = child->get_reachable_hidden_nodes();

            if (reachable_nodes.size() == 0) {
                cout << "\t\tno reachable hidden nodes so cannot split node" << endl;
                continue;
            }

            int r = (rng_float(generator) * reachable_nodes.size());

            CNN_Node *child_node = reachable_nodes[r];
            cout << "\t\tselected node: " << r << " to split with innovation number: " << child_node->get_innovation_number() << endl;

            if (child_node->is_input()) {
                cout << "\t\tchild_node was input, this should never happen!" << endl;
                exit(1);
            }

            if (child_node->is_softmax()) {
                cout << "\t\tchild_node was softmax, this should never happen!" << endl;
                exit(1);
            }

            float depth = child_node->get_depth();
            int size_x = child_node->get_size_x();
            int size_y = child_node->get_size_y();

            CNN_Node *split1 = new CNN_Node(node_innovation_count, depth, batch_size, size_x, size_y, HIDDEN_NODE);
            node_innovation_count++;

            CNN_Node *split2 = new CNN_Node(node_innovation_count, depth, batch_size, size_x, size_y, HIDDEN_NODE);
            node_innovation_count++;

            child->add_node(split1);
            child->add_node(split2);

            vector<CNN_Node*> input_nodes;
            vector<CNN_Node*> output_nodes;

            for (uint32_t i = 0; i < child->get_number_edges(); i++) {
                CNN_Edge *edge = child->get_edge(i);

                if (edge->get_input_innovation_number() == child_node->get_innovation_number()) {
                    CNN_Node *output_node = NULL;
                    for (uint32_t j = 0; j < child->get_number_nodes(); j++) {
                        CNN_Node *node = child->get_node(j);

                        if (node->get_innovation_number() == edge->get_output_innovation_number()) {
                            cout << "\t\tselected output node with innovation number: " << node->get_innovation_number() << endl;
                            output_node = node;
                            break;
                        }
                    }

                    output_nodes.push_back(output_node);
                } else if (edge->get_output_innovation_number() == child_node->get_innovation_number()) {
                    CNN_Node *input_node = NULL;
                    for (uint32_t j = 0; j < child->get_number_nodes(); j++) {
                        CNN_Node *node = child->get_node(j);

                        if (node->get_innovation_number() == edge->get_input_innovation_number()) {
                            cout << "\t\tselected input node with innovation number: " << node->get_innovation_number() << endl;
                            input_node = node;
                            break;
                        }
                    }

                    input_nodes.push_back(input_node);
                }
            }

            int node_1_input_edge_type;
            int node_2_input_edge_type;
            int node_1_output_edge_type;
            int node_2_output_edge_type;

            if (use_sfmp) {
                node_1_input_edge_type = random_edge_type(rng_float(generator));
                node_2_input_edge_type = random_edge_type(rng_float(generator));
                node_1_output_edge_type = random_edge_type(rng_float(generator));
                node_2_output_edge_type = random_edge_type(rng_float(generator));
            } else {
                node_1_input_edge_type = CONVOLUTIONAL;
                node_2_input_edge_type = CONVOLUTIONAL;
                node_1_output_edge_type = CONVOLUTIONAL;
                node_2_output_edge_type = CONVOLUTIONAL;
            }

            //make sure each split node has at least 1 input and 1 output
            int selected_input1 = rng_float(generator) * input_nodes.size();
            int selected_input2 = rng_float(generator) * input_nodes.size();
            int selected_output1 = rng_float(generator) * output_nodes.size();
            int selected_output2 = rng_float(generator) * output_nodes.size();

            add_edge(child, input_nodes[selected_input1], split1, node_1_input_edge_type);
            add_edge(child, input_nodes[selected_input2], split2, node_2_input_edge_type);
            add_edge(child, split1, output_nodes[selected_output1], node_1_output_edge_type);
            add_edge(child, split2, output_nodes[selected_output2], node_2_output_edge_type);

            float input_split_selection_rate = 0.50;
            float output_split_selection_rate = 0.50;

            for (uint32_t i = 0; i < input_nodes.size(); i++) {
                if (i != selected_input1 && rng_float(generator) > input_split_selection_rate) {
                    add_edge(child, input_nodes[i], split1, node_1_input_edge_type);
                }

                if (i != selected_input2 && rng_float(generator) > input_split_selection_rate) {
                    add_edge(child, input_nodes[i], split2, node_2_input_edge_type);
                }
            }

            for (uint32_t i = 0; i < output_nodes.size(); i++) {
                if (i != selected_output1 && rng_float(generator) > output_split_selection_rate) {
                    add_edge(child, split1, output_nodes[i], node_1_output_edge_type);
                }

                if (i != selected_output2 && rng_float(generator) > output_split_selection_rate) {
                    add_edge(child, split2, output_nodes[i], node_2_output_edge_type);
                }
            }
            child_node->disable();

            child->set_generated_by("split_node");
            modifications++;

            continue;
        }
        r -= node_split;

        if (r < node_merge) {
            cout << "\tMERGING A NODE!" << endl;

            //select two nodes
            //  disable edges into and out of those nodes
            //  create new node with depth the average of those two
            //  create edges between new node and all inputs/outputs of merged nodes

            vector<CNN_Node*> reachable_nodes = child->get_reachable_hidden_nodes();

            if (reachable_nodes.size() < 2) {
                cout << "\t\tneed at least two reachable hidden nodes so cannot merge a node" << endl;
                cout << "\t\t\tnumber nodes: " << child->get_number_nodes() << endl;
                cout << "\t\t\tnumber input nodes: " << child->get_number_input_nodes() << endl;
                cout << "\t\t\tnumber softmax nodes: " << child->get_number_softmax_nodes() << endl;
                continue;
            }

            int r1 = rng_float(generator) * reachable_nodes.size();
            int r2 = rng_float(generator) * (reachable_nodes.size() - 1);

            //will select two distinct nodes
            if (r1 == r2) r2++;

            cout << "\t\tr1: " << r1 << ", r2: " << r2 << endl;

            CNN_Node *node1 = reachable_nodes[r1];
            cout << "\t\tselected node: " << r1 << " with innovation number: " << node1->get_innovation_number() << endl;

            CNN_Node *node2 = reachable_nodes[r2];
            cout << "\t\tselected node: " << r2 << " with innovation number: " << node2->get_innovation_number() << endl;

            if (node1->is_input()) {
                cout << "\t\tnode1 was input, this should never happen!" << endl;
                exit(1);
            }

            if (node1->is_softmax()) {
                cout << "\t\tnode1 was softmax, this should never happen!" << endl;
                exit(1);
            }

            if (node2->is_input()) {
                cout << "\t\tnode2 was input, this should never happen!" << endl;
                exit(1);
            }

            if (node2->is_softmax()) {
                cout << "\t\tnode2 was softmax, this should never happen!" << endl;
                exit(1);
            }

            float depth = (node1->get_depth() + node2->get_depth()) / 2.0;
            int size_x = (node1->get_size_x() + node2->get_size_x()) / 2.0;
            int size_y = (node1->get_size_y() + node2->get_size_y()) / 2.0;

            CNN_Node *merged_node = new CNN_Node(node_innovation_count, depth, batch_size, size_x, size_y, HIDDEN_NODE);
            node_innovation_count++;

            child->add_node(merged_node);

            vector<CNN_Node*> connected_nodes;
            for (uint32_t i = 0; i < child->get_number_edges(); i++) {
                CNN_Edge *edge = child->get_edge(i);

                if (edge->get_input_innovation_number() == node1->get_innovation_number() ||
                        edge->get_input_innovation_number() == node2->get_innovation_number()) {

                    edge->disable();

                    for (uint32_t j = 0; j < child->get_number_nodes(); j++) {
                        CNN_Node *node = child->get_node(j);

                        if (node->get_innovation_number() == edge->get_output_innovation_number()) {
                            connected_nodes.push_back(node);
                            break;
                        }
                    }
                }

                if (edge->get_output_innovation_number() == node1->get_innovation_number() ||
                        edge->get_output_innovation_number() == node2->get_innovation_number()) {

                    edge->disable();

                    for (uint32_t j = 0; j < child->get_number_nodes(); j++) {
                        CNN_Node *node = child->get_node(j);

                        if (node->get_innovation_number() == edge->get_input_innovation_number()) {
                            connected_nodes.push_back(node);
                            break;
                        }
                    }
                }
            }

            int input_edge_type;
            int output_edge_type;

            if (use_sfmp) {
                input_edge_type = random_edge_type(rng_float(generator));
                output_edge_type = random_edge_type(rng_float(generator));
            } else {
                input_edge_type = CONVOLUTIONAL;
                output_edge_type = CONVOLUTIONAL;
            }

            for (uint32_t i = 0; i < connected_nodes.size(); i++) {
                if (connected_nodes[i]->get_depth() < depth) {
                    add_edge(child, connected_nodes[i], merged_node, input_edge_type);
                } else if (connected_nodes[i]->get_depth() > depth) {
                    add_edge(child, merged_node, connected_nodes[i], output_edge_type);
                }
            }

            //disable the merged nodes so they can later be enabled
            node1->disable();
            node2->disable();

            child->set_generated_by("merge_node");
            modifications++;

            continue;
        }
        r -= node_merge;


        if (r < node_enable) {
            cout << "\tENABLING NODE!" << endl;

            vector<CNN_Node*> disabled_nodes = child->get_disabled_nodes();

            if (disabled_nodes.size() == 0) {
                cout << "\t\tThere were no disabled hidden nodes, skipping!" << endl;
                continue;
            }

            int r = rng_float(generator) * disabled_nodes.size();

            CNN_Node *node = disabled_nodes[r];
            node->enable();

            for (uint32_t i = 0; i < child->get_number_edges(); i++) {
                CNN_Edge *edge = child->get_edge(i);

                if (edge->get_output_innovation_number() == node->get_innovation_number() || edge->get_input_innovation_number() == node->get_innovation_number()) {
                    edge->enable();
                    //reinitialize weights for re-enabled edge
                    edge->set_needs_init();
                }
            }

            child->set_generated_by("enable_node");
            modifications++;

            continue;
        }
        r -= node_enable;

        if (r < node_disable) {
            cout << "\tDISABLING NODE!" << endl;

            vector<CNN_Node*> enabled_nodes = child->get_reachable_hidden_nodes();

            if (enabled_nodes.size() == 0) {
                cout << "\t\tThere were no enabled hidden nodes, skipping!" << endl;
                continue;
            }

            int r = rng_float(generator) * enabled_nodes.size();

            CNN_Node *node = enabled_nodes[r];
            node->disable();

            for (uint32_t i = 0; i < child->get_number_edges(); i++) {
                CNN_Edge *edge = child->get_edge(i);

                if (edge->get_output_innovation_number() == node->get_innovation_number() || edge->get_input_innovation_number() == node->get_innovation_number()) {
                    edge->disable();
                }
            }

            child->set_generated_by("disable_node");
            modifications++;

            continue;
        }
        r -= node_disable;

        cout << "ERROR: problem choosing mutation type -- should never get here!" << endl;
        cout << "\tremaining random value (for mutation selection): " << r << endl;
        exit(1);
    }

    return child;
}


bool edges_contains(vector< CNN_Edge* > &edges, CNN_Edge *edge) {
    for (int32_t i = 0; i < (int32_t)edges.size(); i++) {
        if (edges[i]->get_innovation_number() == edge->get_innovation_number()) return true;
    }
    return false;
}

CNN_Edge* attempt_edge_insert(vector<CNN_Edge*> &child_edges, CNN_Edge *edge) {
    for (int32_t i = 0; i < (int32_t)child_edges.size(); i++) {
        if (child_edges[i]->get_innovation_number() == edge->get_innovation_number()) {
            cerr << "ERROR in crossover! trying to push an edge with innovation_number: " << edge->get_innovation_number() << " and it already exists in the vector!" << endl;
            /*
            cerr << "p1_position: " << p1_position << ", p1_size: " << p1_child_edges.size() << endl;
            cerr << "p2_position: " << p2_position << ", p2_size: " << p2_child_edges.size() << endl;
            cerr << "vector innovation numbers: " << endl;
            */
            for (int32_t i = 0; i < (int32_t)child_edges.size(); i++) {
                cerr << "\t" << child_edges[i]->get_innovation_number() << endl;
            }

            cerr << "This should never happen!" << endl;
            exit(1);

            return NULL;
        } else if (child_edges[i]->get_input_innovation_number() == edge->get_input_innovation_number() &&
                child_edges[i]->get_output_innovation_number() == edge->get_output_innovation_number()) {

            cerr << "Not inserting edge in crossover operation as there was already an edge with the same input and output innovation numbers!" << endl;
            return NULL;
        }
    }

    //edges have already been copied
    child_edges.insert( upper_bound(child_edges.begin(), child_edges.end(), edge, sort_CNN_Edges_by_depth()), edge);

    return edge;
}

void attempt_node_insert(vector<CNN_Node*> &child_nodes, CNN_Node *node) {
    for (int32_t i = 0; i < (int32_t)child_nodes.size(); i++) {
        if (child_nodes[i]->get_innovation_number() == node->get_innovation_number()) return;
    }

    CNN_Node *node_copy = node->copy();
    child_nodes.insert( upper_bound(child_nodes.begin(), child_nodes.end(), node_copy, sort_CNN_Nodes_by_depth()), node_copy);
}

CNN_Genome* EXACT::create_child() {
    cout << "\tCREATING CHILD THROUGH CROSSOVER!" << endl;
    int r1 = rng_float(generator) * genomes.size();
    int r2 = rng_float(generator) * (genomes.size() - 1);
    if (r1 >= r2) r2++;

    //parent should have higher fitness
    if (r2 < r1) {
        int tmp = r2;
        r2 = r1;
        r1 = tmp;
    }
    cout << "\t\tparent positions: " << r1 << " and " << r2 << endl;

    CNN_Genome *parent1 = genomes[r1];
    CNN_Genome *parent2 = genomes[r2];

    cout << "\t\tgenerating child " << genomes_generated << " from parents: " << parent1->get_generation_id() << " and " << parent2->get_generation_id() << endl;

    //nodes are copied in the attempt_node_insert_function
    vector< CNN_Node* > child_nodes;
    vector< CNN_Edge* > child_edges;

    int p1_position = 0;
    int p2_position = 0;

    //edges are not sorted in order of innovation number, they need to be
    vector< CNN_Edge* > p1_edges;
    vector< CNN_Edge* > p2_edges;
    parent1->get_edge_copies(p1_edges);
    parent2->get_edge_copies(p2_edges);

    sort(p1_edges.begin(), p1_edges.end(), sort_CNN_Edges_by_innovation());
    sort(p2_edges.begin(), p2_edges.end(), sort_CNN_Edges_by_innovation());

    /*
    cerr << "p1 innovation numbers AFTER SORT: " << endl;
    for (int32_t i = 0; i < (int32_t)p1_edges.size(); i++) {
        cerr << "\t" << p1_edges[i]->get_innovation_number() << endl;
    }
    cerr << "p2 innovation numbers AFTER SORT: " << endl;
    for (int32_t i = 0; i < (int32_t)p2_edges.size(); i++) {
        cerr << "\t" << p2_edges[i]->get_innovation_number() << endl;
    }
    */

    while (p1_position < (int32_t)p1_edges.size() && p2_position < (int32_t)p2_edges.size()) {
        CNN_Edge* p1_edge = p1_edges[p1_position];
        CNN_Edge* p2_edge = p2_edges[p2_position];

        int p1_innovation = p1_edge->get_innovation_number();
        int p2_innovation = p2_edge->get_innovation_number();

        if (p1_innovation == p2_innovation) {
            CNN_Edge *inserted_edge = NULL;
            if ((inserted_edge = attempt_edge_insert(child_edges, p1_edge)) != NULL) {
                //push back surrounding nodes
                attempt_node_insert(child_nodes, p1_edge->get_input_node());
                attempt_node_insert(child_nodes, p1_edge->get_output_node());
            }

            p1_position++;
            p2_position++;
        } else if (p1_innovation < p2_innovation) {
            CNN_Edge *inserted_edge = NULL;
            if ((inserted_edge = attempt_edge_insert(child_edges, p1_edge)) != NULL) {

                if (rng_float(generator) <= more_fit_parent_crossover) {
                    inserted_edge->disable();
                } else if (rng_float(generator) <= crossover_alter_edge_type) {
                    inserted_edge->alter_edge_type();
                }

                //push back surrounding nodes
                attempt_node_insert(child_nodes, p1_edge->get_input_node());
                attempt_node_insert(child_nodes, p1_edge->get_output_node());
            }

            p1_position++;
        } else {
            CNN_Edge *inserted_edge = NULL;
            if ((inserted_edge = attempt_edge_insert(child_edges, p2_edge)) != NULL) {

                if (rng_float(generator) >= less_fit_parent_crossover) {
                    inserted_edge->disable();
                } else if (rng_float(generator) <= crossover_alter_edge_type) {
                    inserted_edge->alter_edge_type();
                }

                //push back surrounding nodes
                attempt_node_insert(child_nodes, p2_edge->get_input_node());
                attempt_node_insert(child_nodes, p2_edge->get_output_node());
            }

            p2_position++;
        }
    }

    while (p1_position < (int32_t)p1_edges.size()) {
        CNN_Edge* p1_edge = p1_edges[p1_position];

        CNN_Edge *inserted_edge = NULL;
        if ((inserted_edge = attempt_edge_insert(child_edges, p1_edge)) != NULL) {

            if (rng_float(generator) >= more_fit_parent_crossover) {
                inserted_edge->disable();
            }

            //push back surrounding nodes
            attempt_node_insert(child_nodes, p1_edge->get_input_node());
            attempt_node_insert(child_nodes, p1_edge->get_output_node());
        }

        p1_position++;
    }

    while (p2_position < (int32_t)p2_edges.size()) {
        CNN_Edge* p2_edge = p2_edges[p2_position];

        CNN_Edge *inserted_edge = NULL;
        if ((inserted_edge = attempt_edge_insert(child_edges, p2_edge)) != NULL) {
            if (rng_float(generator) >= less_fit_parent_crossover) {
                inserted_edge->disable();
            }

            //push back surrounding nodes
            attempt_node_insert(child_nodes, p2_edge->get_input_node());
            attempt_node_insert(child_nodes, p2_edge->get_output_node());
        }

        p2_position++;
    }

    sort(child_edges.begin(), child_edges.end(), sort_CNN_Edges_by_depth());
    sort(child_nodes.begin(), child_nodes.end(), sort_CNN_Nodes_by_depth());

    for (int32_t i = 0; i < (int32_t)child_edges.size(); i++) {
        if (!child_edges[i]->set_nodes(child_nodes)) {
            cout << "\t\treinitializing weights of copy" << endl;
            child_edges[i]->resize();
        }
    }

    long genome_seed = rng_long(generator);

    float mu, mu_delta, learning_rate, learning_rate_delta, weight_decay, weight_decay_delta, alpha, input_dropout_probability, hidden_dropout_probability;
    int velocity_reset, batch_size;

    if (inserted_genomes < (population_size * 10)) {
        cout << "\tGenerating hyperparameters randomly." << endl;
        generate_initial_hyperparameters(mu, mu_delta, learning_rate, learning_rate_delta, weight_decay, weight_decay_delta, alpha, velocity_reset, input_dropout_probability, hidden_dropout_probability, batch_size);
    } else {
        cout << "\tGenerating hyperparameters with simplex." << endl;
        generate_simplex_hyperparameters(mu, mu_delta, learning_rate, learning_rate_delta, weight_decay, weight_decay_delta, alpha, velocity_reset, input_dropout_probability, hidden_dropout_probability, batch_size);
    }

    CNN_Genome *child = new CNN_Genome(genomes_generated++, padding, number_training_images, number_validation_images, number_test_images, genome_seed, max_epochs, reset_weights, velocity_reset, mu, mu_delta, learning_rate, learning_rate_delta, weight_decay, weight_decay_delta, batch_size, epsilon, alpha, input_dropout_probability, hidden_dropout_probability, child_nodes, child_edges);

    child->set_generated_by("crossover");

    return child;
}


void EXACT::write_individual_hyperparameters(CNN_Genome *individual) {
    fstream out(output_directory + "/individual_hyperparameters.txt", fstream::out | fstream::app);

    out << individual->get_best_validation_error()
        << "," << individual->get_best_validation_error()
        << "," << individual->get_test_error()
        << "," << individual->get_number_edges()
        << "," << individual->get_number_nodes()
        << "," << individual->get_number_weights()

        << "," << individual->get_initial_mu()
        << "," << individual->get_mu()
        << "," << individual->get_mu_delta()
        << "," << individual->get_initial_learning_rate()
        << "," << individual->get_learning_rate()
        << "," << individual->get_learning_rate_delta()
        << "," << individual->get_initial_weight_decay()
        << "," << individual->get_weight_decay()
        << "," << individual->get_weight_decay_delta()
        << "," << individual->get_alpha()
        << "," << individual->get_input_dropout_probability()
        << "," << individual->get_hidden_dropout_probability()
        << "," << individual->get_batch_size()
        << "," << individual->get_velocity_reset()

        << endl;

    out.close();
}

void EXACT::write_statistics(int new_generation_id, float new_fitness) {
    float min_fitness = EXACT_MAX_FLOAT;
    float max_fitness = -EXACT_MAX_FLOAT;
    float avg_fitness = 0.0;
    int fitness_count = 0;

    float min_epochs = EXACT_MAX_FLOAT;
    float max_epochs = -EXACT_MAX_FLOAT;
    float avg_epochs = 0.0;
    int epochs_count = 0;

    float min_nodes = EXACT_MAX_FLOAT;
    float max_nodes = -EXACT_MAX_FLOAT;
    float avg_nodes = 0.0;
    int nodes_count = 0;

    float min_pooling_edges = EXACT_MAX_FLOAT;
    float max_pooling_edges = -EXACT_MAX_FLOAT;
    float avg_pooling_edges = 0.0;
    int pooling_edges_count = 0;

    float min_convolutional_edges = EXACT_MAX_FLOAT;
    float max_convolutional_edges = -EXACT_MAX_FLOAT;
    float avg_convolutional_edges = 0.0;
    int convolutional_edges_count = 0;

    float min_weights = EXACT_MAX_FLOAT;
    float max_weights = -EXACT_MAX_FLOAT;
    float avg_weights = 0.0;
    int weights_count = 0;


    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        float fitness = genomes[i]->get_best_validation_error();

        if (fitness != EXACT_MAX_FLOAT) {
            avg_fitness += fitness;
            fitness_count++;
        }

        if (fitness < min_fitness) min_fitness = fitness;
        if (fitness > max_fitness) max_fitness = fitness;

        float epochs = genomes[i]->get_best_epoch();

        if (epochs != EXACT_MAX_FLOAT) {
            avg_epochs += epochs;
            epochs_count++;
        }

        if (epochs < min_epochs) min_epochs = epochs;
        if (epochs > max_epochs) max_epochs = epochs;

        float nodes = genomes[i]->get_number_enabled_nodes();

        if (nodes != EXACT_MAX_FLOAT) {
            avg_nodes += nodes;
            nodes_count++;
        }

        if (nodes < min_nodes) min_nodes = nodes;
        if (nodes > max_nodes) max_nodes = nodes;

        float pooling_edges = genomes[i]->get_number_enabled_pooling_edges();

        if (pooling_edges != EXACT_MAX_FLOAT) {
            avg_pooling_edges += pooling_edges;
            pooling_edges_count++;
        }

        if (pooling_edges < min_pooling_edges) min_pooling_edges = pooling_edges;
        if (pooling_edges > max_pooling_edges) max_pooling_edges = pooling_edges;

        float convolutional_edges = genomes[i]->get_number_enabled_convolutional_edges();

        if (convolutional_edges != EXACT_MAX_FLOAT) {
            avg_convolutional_edges += convolutional_edges;
            convolutional_edges_count++;
        }

        if (convolutional_edges < min_convolutional_edges) min_convolutional_edges = convolutional_edges;
        if (convolutional_edges > max_convolutional_edges) max_convolutional_edges = convolutional_edges;


        float weights = genomes[i]->get_number_weights();

        if (weights != EXACT_MAX_FLOAT) {
            avg_weights += weights;
            weights_count++;
        }

        if (weights < min_weights) min_weights = weights;
        if (weights > max_weights) max_weights = weights;
    }
    avg_fitness /= fitness_count;
    avg_epochs /= epochs_count;
    avg_nodes /= nodes_count;
    avg_pooling_edges /= pooling_edges_count;
    avg_convolutional_edges /= convolutional_edges_count;
    avg_weights /= weights_count;

    if (fitness_count == 0) avg_fitness = 0.0;
    if (min_fitness == EXACT_MAX_FLOAT) min_fitness = 0;
    if (max_fitness == EXACT_MAX_FLOAT) max_fitness = 0;
    if (max_fitness == -EXACT_MAX_FLOAT) max_fitness = 0;

    if (epochs_count == 0) avg_epochs = 0.0;
    if (min_epochs == EXACT_MAX_FLOAT) min_epochs = 0;
    if (max_epochs == EXACT_MAX_FLOAT) max_epochs = 0;
    if (max_epochs == -EXACT_MAX_FLOAT) max_epochs = 0;

    if (nodes_count == 0) avg_nodes = 0.0;
    if (min_nodes == EXACT_MAX_FLOAT) min_nodes = 0;
    if (max_nodes == EXACT_MAX_FLOAT) max_nodes = 0;
    if (max_nodes == -EXACT_MAX_FLOAT) max_nodes = 0;

    if (pooling_edges_count == 0) avg_pooling_edges = 0.0;
    if (min_pooling_edges == EXACT_MAX_FLOAT) min_pooling_edges = 0;
    if (max_pooling_edges == EXACT_MAX_FLOAT) max_pooling_edges = 0;
    if (max_pooling_edges == -EXACT_MAX_FLOAT) max_pooling_edges = 0;

    if (convolutional_edges_count == 0) avg_convolutional_edges = 0.0;
    if (min_convolutional_edges == EXACT_MAX_FLOAT) min_convolutional_edges = 0;
    if (max_convolutional_edges == EXACT_MAX_FLOAT) max_convolutional_edges = 0;
    if (max_convolutional_edges == -EXACT_MAX_FLOAT) max_convolutional_edges = 0;


    if (weights_count == 0) avg_weights = 0.0;
    if (min_weights == EXACT_MAX_FLOAT) min_weights = 0;
    if (max_weights == EXACT_MAX_FLOAT) max_weights = 0;
    if (max_weights == -EXACT_MAX_FLOAT) max_weights = 0;


    fstream out(output_directory + "/progress.txt", fstream::out | fstream::app);

    out << setw(16) << time(NULL)
        << setw(16) << new_generation_id
        << setw(16) << new_fitness
        << setw(16) << inserted_genomes
        << setw(16) << setprecision(5) << fixed << min_fitness
        << setw(16) << setprecision(5) << fixed << avg_fitness
        << setw(16) << setprecision(5) << fixed << max_fitness
        << setw(16) << setprecision(5) << fixed << min_epochs
        << setw(16) << setprecision(5) << fixed << avg_epochs
        << setw(16) << setprecision(5) << fixed << max_epochs
        << setw(16) << setprecision(5) << fixed << min_nodes
        << setw(16) << setprecision(5) << fixed << avg_nodes
        << setw(16) << setprecision(5) << fixed << max_nodes
        << setw(16) << setprecision(5) << fixed << min_pooling_edges
        << setw(16) << setprecision(5) << fixed << avg_pooling_edges
        << setw(16) << setprecision(5) << fixed << max_pooling_edges
        << setw(16) << setprecision(5) << fixed << min_convolutional_edges
        << setw(16) << setprecision(5) << fixed << avg_convolutional_edges
        << setw(16) << setprecision(5) << fixed << max_convolutional_edges
        << setw(16) << setprecision(5) << fixed << min_weights
        << setw(16) << setprecision(5) << fixed << avg_weights
        << setw(16) << setprecision(5) << fixed << max_weights;

    for (auto i = generated_from_map.begin(); i != generated_from_map.end(); i++) {
        if (generated_from_map[i->first] == 0) {
            out << setw(16) << setprecision(3) << 0.0;
        } else {
            out << setw(16) << setprecision(3) << (100.0 * (float)inserted_from_map[i->first] / (float)generated_from_map[i->first]);
        }
    }
    out << endl;

    out.close();

    out = fstream(output_directory + "/hyperparameters.txt", fstream::out | fstream::app);

    float min_initial_mu = 10, max_initial_mu = 0, avg_initial_mu = 0;
    float min_mu_delta = 10, max_mu_delta = 0, avg_mu_delta = 0;

    float min_initial_learning_rate = 10, max_initial_learning_rate = 0, avg_initial_learning_rate = 0;
    float min_learning_rate_delta = 10, max_learning_rate_delta = 0, avg_learning_rate_delta = 0;

    float min_initial_weight_decay = 10, max_initial_weight_decay = 0, avg_initial_weight_decay = 0;
    float min_weight_decay_delta = 10, max_weight_decay_delta = 0, avg_weight_decay_delta = 0;

    float min_velocity_reset = 10000000, max_velocity_reset = 0, avg_velocity_reset = 0;

    float min_alpha = 10000000, max_alpha = 0, avg_alpha = 0;

    float min_input_dropout_probability = 10000000, max_input_dropout_probability = 0, avg_input_dropout_probability = 0;
    float min_hidden_dropout_probability = 10000000, max_hidden_dropout_probability = 0, avg_hidden_dropout_probability = 0;

    float min_batch_size = 10000000, max_batch_size = 0, avg_batch_size = 0;

    float best_initial_mu = genomes[0]->get_initial_mu();
    float best_mu_delta = genomes[0]->get_mu_delta();

    float best_initial_learning_rate = genomes[0]->get_initial_learning_rate();
    float best_learning_rate_delta = genomes[0]->get_learning_rate_delta();

    float best_initial_weight_decay = genomes[0]->get_initial_weight_decay();
    float best_weight_decay_delta = genomes[0]->get_weight_decay_delta();

    float best_alpha = genomes[0]->get_alpha();

    float best_velocity_reset = genomes[0]->get_velocity_reset();

    float best_input_dropout_probability = genomes[0]->get_input_dropout_probability();
    float best_hidden_dropout_probability = genomes[0]->get_hidden_dropout_probability();

    float best_batch_size = genomes[0]->get_batch_size();



    for (uint32_t i = 0; i < genomes.size(); i++) {
        if (genomes[i]->get_initial_mu() < min_initial_mu) {
            min_initial_mu = genomes[i]->get_initial_mu();
        }

        if (genomes[i]->get_initial_mu() > max_initial_mu) {
            max_initial_mu = genomes[i]->get_initial_mu();
        }
        avg_initial_mu += genomes[i]->get_initial_mu();

        if (genomes[i]->get_mu_delta() < min_mu_delta) {
            min_mu_delta = genomes[i]->get_mu_delta();
        }

        if (genomes[i]->get_mu_delta() > max_mu_delta) {
            max_mu_delta = genomes[i]->get_mu_delta();
        }
        avg_mu_delta += genomes[i]->get_mu_delta();


        if (genomes[i]->get_initial_learning_rate() < min_initial_learning_rate) {
            min_initial_learning_rate = genomes[i]->get_initial_learning_rate();
        }

        if (genomes[i]->get_initial_learning_rate() > max_initial_learning_rate) {
            max_initial_learning_rate = genomes[i]->get_initial_learning_rate();
        }
        avg_initial_learning_rate += genomes[i]->get_initial_learning_rate();

        if (genomes[i]->get_learning_rate_delta() < min_learning_rate_delta) {
            min_learning_rate_delta = genomes[i]->get_learning_rate_delta();
        }

        if (genomes[i]->get_learning_rate_delta() > max_learning_rate_delta) {
            max_learning_rate_delta = genomes[i]->get_learning_rate_delta();
        }
        avg_learning_rate_delta += genomes[i]->get_learning_rate_delta();


        if (genomes[i]->get_initial_weight_decay() < min_initial_weight_decay) {
            min_initial_weight_decay = genomes[i]->get_initial_weight_decay();
        }

        if (genomes[i]->get_initial_weight_decay() > max_initial_weight_decay) {
            max_initial_weight_decay = genomes[i]->get_initial_weight_decay();
        }
        avg_initial_weight_decay += genomes[i]->get_initial_weight_decay();

        if (genomes[i]->get_weight_decay_delta() < min_weight_decay_delta) {
            min_weight_decay_delta = genomes[i]->get_weight_decay_delta();
        }

        if (genomes[i]->get_weight_decay_delta() > max_weight_decay_delta) {
            max_weight_decay_delta = genomes[i]->get_weight_decay_delta();
        }
        avg_weight_decay_delta += genomes[i]->get_weight_decay_delta();


        if (genomes[i]->get_alpha() < min_alpha) {
            min_alpha = genomes[i]->get_alpha();
        }

        if (genomes[i]->get_alpha() > max_alpha) {
            max_alpha = genomes[i]->get_alpha();
        }
        avg_alpha += genomes[i]->get_alpha();


        if (genomes[i]->get_velocity_reset() < min_velocity_reset) {
            min_velocity_reset = genomes[i]->get_velocity_reset();
        }

        if (genomes[i]->get_velocity_reset() > max_velocity_reset) {
            max_velocity_reset = genomes[i]->get_velocity_reset();
        }
        avg_velocity_reset += genomes[i]->get_velocity_reset();


        if (genomes[i]->get_input_dropout_probability() < min_input_dropout_probability) {
            min_input_dropout_probability = genomes[i]->get_input_dropout_probability();
        }

        if (genomes[i]->get_input_dropout_probability() > max_input_dropout_probability) {
            max_input_dropout_probability = genomes[i]->get_input_dropout_probability();
        }
        avg_input_dropout_probability += genomes[i]->get_input_dropout_probability();

        if (genomes[i]->get_hidden_dropout_probability() < min_hidden_dropout_probability) {
            min_hidden_dropout_probability = genomes[i]->get_hidden_dropout_probability();
        }

        if (genomes[i]->get_hidden_dropout_probability() > max_hidden_dropout_probability) {
            max_hidden_dropout_probability = genomes[i]->get_hidden_dropout_probability();
        }
        avg_hidden_dropout_probability += genomes[i]->get_hidden_dropout_probability();

        if (genomes[i]->get_batch_size() < min_batch_size) {
            min_batch_size = genomes[i]->get_batch_size();
        }

        if (genomes[i]->get_batch_size() > max_batch_size) {
            max_batch_size = genomes[i]->get_batch_size();
        }
        avg_batch_size += genomes[i]->get_batch_size();
    }

    avg_initial_mu /= genomes.size();
    avg_mu_delta /= genomes.size();

    avg_initial_learning_rate /= genomes.size();
    avg_learning_rate_delta /= genomes.size();

    avg_initial_weight_decay /= genomes.size();
    avg_weight_decay_delta /= genomes.size();

    avg_alpha /= genomes.size();

    avg_velocity_reset /= genomes.size();

    avg_input_dropout_probability /= genomes.size();
    avg_hidden_dropout_probability /= genomes.size();

    avg_batch_size /= genomes.size();

    out << setw(20) << setprecision(11) << min_initial_mu
        << setw(20) << setprecision(11) << max_initial_mu
        << setw(20) << setprecision(11) << avg_initial_mu
        << setw(20) << setprecision(11) << best_initial_mu
        << setw(20) << setprecision(11) << min_mu_delta
        << setw(20) << setprecision(11) << max_mu_delta
        << setw(20) << setprecision(11) << avg_mu_delta
        << setw(20) << setprecision(11) << best_mu_delta

        << setw(20) << setprecision(11) << min_initial_learning_rate
        << setw(20) << setprecision(11) << max_initial_learning_rate
        << setw(20) << setprecision(11) << avg_initial_learning_rate
        << setw(20) << setprecision(11) << best_initial_learning_rate
        << setw(20) << setprecision(11) << min_learning_rate_delta
        << setw(20) << setprecision(11) << max_learning_rate_delta
        << setw(20) << setprecision(11) << avg_learning_rate_delta
        << setw(20) << setprecision(11) << best_learning_rate_delta

        << setw(20) << setprecision(11) << min_initial_weight_decay
        << setw(20) << setprecision(11) << max_initial_weight_decay
        << setw(20) << setprecision(11) << avg_initial_weight_decay
        << setw(20) << setprecision(11) << best_initial_weight_decay
        << setw(20) << setprecision(11) << min_weight_decay_delta
        << setw(20) << setprecision(11) << max_weight_decay_delta
        << setw(20) << setprecision(11) << avg_weight_decay_delta
        << setw(20) << setprecision(11) << best_weight_decay_delta

        << setw(20) << setprecision(11) << min_alpha
        << setw(20) << setprecision(11) << max_alpha
        << setw(20) << setprecision(11) << avg_alpha
        << setw(20) << setprecision(11) << best_alpha

        << setw(20) << setprecision(11) << min_velocity_reset
        << setw(20) << setprecision(11) << max_velocity_reset
        << setw(20) << setprecision(11) << avg_velocity_reset
        << setw(20) << setprecision(11) << best_velocity_reset

        << setw(20) << setprecision(11) << min_input_dropout_probability
        << setw(20) << setprecision(11) << max_input_dropout_probability
        << setw(20) << setprecision(11) << avg_input_dropout_probability
        << setw(20) << setprecision(11) << best_input_dropout_probability
        << setw(20) << setprecision(11) << min_hidden_dropout_probability
        << setw(20) << setprecision(11) << max_hidden_dropout_probability
        << setw(20) << setprecision(11) << avg_hidden_dropout_probability
        << setw(20) << setprecision(11) << best_hidden_dropout_probability

        << setw(20) << setprecision(11) << min_batch_size
        << setw(20) << setprecision(11) << max_batch_size
        << setw(20) << setprecision(11) << avg_batch_size
        << setw(20) << setprecision(11) << best_batch_size

        << endl;
 
    out.close();
}

void EXACT::write_hyperparameters_header() {
    ifstream f(output_directory + "/hyperparameters.txt");
    if (f.good()) return;   //return if file already exists, don't need to rewrite header

    fstream out(output_directory + "/hyperparameters.txt", fstream::out | fstream::app);
    out << "# min initial mu"
        << ", max initial mu"
        << ", avg initial mu"
        << ", best initial mu"
        << ", min mu delta"
        << ", max mu delta"
        << ", avg mu delta"
        << ", best mu delta"

        << ", min initial learning rate"
        << ", max initial learning rate"
        << ", avg initial learning rate"
        << ", best initial learning rate"
        << ", min learning rate delta"
        << ", max learning rate delta"
        << ", avg learning rate delta"
        << ", best learning rate delta"

        << ", min initial weight decay"
        << ", max initial weight decay"
        << ", avg initial weight decay"
        << ", best initial weight decay"
        << ", min weight decay delta"
        << ", max weight decay delta"
        << ", avg weight decay delta"
        << ", best weight decay delta"

        << ", min alpha"
        << ", max alpha"
        << ", avg alpha"
        << ", best alpha"

        << ", min velocity reset"
        << ", max velocity reset"
        << ", avg velocity reset"
        << ", best velocity reset"

        << ", min input dropout probability"
        << ", max input dropout probability"
        << ", avg input dropout probability"
        << ", best input dropout probability"
        << ", min hidden dropout probability"
        << ", max hidden dropout probability"
        << ", avg hidden dropout probability"
        << ", best hidden dropout probability"

        << ", min batch size"
        << ", max batch size"
        << ", avg batch size"
        << ", best batch size"


        << endl;

    out.close();
}

void EXACT::write_statistics_header() {
    ifstream f(output_directory + "/progress.txt");
    if (f.good()) return;   //return if file already exists, don't need to rewrite header

    fstream out(output_directory + "/progress.txt", fstream::out | fstream::app);
    out << "# " << setw(14) << "time"
        << ", " << setw(14) << "generation id"
        << ", " << setw(14) << "new fitness"
        << ", " << setw(14) << "inserted"
        << ", " << setw(14) << "min_fitness"
        << ", " << setw(14) << "avg_fitness"
        << ", " << setw(14) << "max_fitness"
        << ", " << setw(14) << "min_epochs"
        << ", " << setw(14) << "avg_epochs"
        << ", " << setw(14) << "max_epochs"
        << ", " << setw(14) << "min_nodes"
        << ", " << setw(14) << "avg_nodes"
        << ", " << setw(14) << "max_nodes"
        << ", " << setw(14) << "min_pooling_edges"
        << ", " << setw(14) << "avg_pooling_edges"
        << ", " << setw(14) << "max_pooling_edges"
        << ", " << setw(14) << "min_convolutional_edges"
        << ", " << setw(14) << "avg_convolutional_edges"
        << ", " << setw(14) << "max_convolutional_edges"
        << ", " << setw(14) << "min_weights"
        << ", " << setw(14) << "avg_weights"
        << ", " << setw(14) << "max_weights";

    for (auto i = generated_from_map.begin(); i != generated_from_map.end(); i++) {
        out << ", " << setw(14) << i->first;
    }
    out << endl;

    out.close();
}


bool EXACT::is_identical(EXACT *other, bool testing_checkpoint) {
    if (are_different("id", id, other->id)) return false;

    if (are_different("search_name", search_name, other->search_name)) return false;
    if (are_different("output_directory", output_directory, other->output_directory)) return false;
    if (are_different("training_filename", training_filename, other->training_filename)) return false;
    if (are_different("validation_filename", validation_filename, other->validation_filename)) return false;
    if (are_different("test_filename", test_filename, other->test_filename)) return false;

    if (are_different("number_training_images", number_training_images, other->number_training_images)) return false;
    if (are_different("number_validation_images", number_validation_images, other->number_validation_images)) return false;
    if (are_different("number_test_images", number_test_images, other->number_test_images)) return false;

    if (are_different("padding", padding, other->padding)) return false;

    if (are_different("image_channels", image_channels, other->image_channels)) return false;
    if (are_different("image_rows", image_rows, other->image_rows)) return false;
    if (are_different("image_cols", image_cols, other->image_cols)) return false;
    if (are_different("number_classes", number_classes, other->number_classes)) return false;

    if (are_different("population_size", population_size, other->population_size)) return false;
    if (are_different("node_innovation_count", node_innovation_count, other->node_innovation_count)) return false;
    if (are_different("edge_innovation_count", edge_innovation_count, other->edge_innovation_count)) return false;

    if (are_different("generator", generator, other->generator)) return false;
    if (are_different("normal_distribution", normal_distribution, other->normal_distribution)) return false;
    if (are_different("rng_long", rng_long, other->rng_long)) return false;
    if (are_different("rng_float", rng_float, other->rng_float)) return false;

    if (are_different("genomes_generated", genomes_generated, other->genomes_generated)) return false;
    if (are_different("inserted_genomes", inserted_genomes, other->inserted_genomes)) return false;
    if (are_different("max_genomes", max_genomes, other->max_genomes)) return false;

    if (are_different("reset_weights", reset_weights, other->reset_weights)) return false;
    if (are_different("max_epochs", max_epochs, other->max_epochs)) return false;
    if (are_different("use_sfmp", use_sfmp, other->use_sfmp)) return false;
    if (are_different("use_node_operations", use_node_operations, other->use_node_operations)) return false;

    if (are_different("initial_batch_size_min", initial_batch_size_min, other->initial_batch_size_min)) return false;
    if (are_different("initial_batch_size_max", initial_batch_size_max, other->initial_batch_size_max)) return false;
    if (are_different("batch_size_min", batch_size_min, other->batch_size_min)) return false;
    if (are_different("batch_size_max", batch_size_max, other->batch_size_max)) return false;

    if (are_different("initial_mu_min", initial_mu_min, other->initial_mu_min)) return false;
    if (are_different("initial_mu_max", initial_mu_max, other->initial_mu_max)) return false;
    if (are_different("mu_min", mu_min, other->mu_min)) return false;
    if (are_different("mu_max", mu_max, other->mu_max)) return false;

    if (are_different("initial_mu_delta_min", initial_mu_delta_min, other->initial_mu_delta_min)) return false;
    if (are_different("initial_mu_delta_max", initial_mu_delta_max, other->initial_mu_delta_max)) return false;
    if (are_different("mu_delta_min", mu_delta_min, other->mu_delta_min)) return false;
    if (are_different("mu_delta_max", mu_delta_max, other->mu_delta_max)) return false;

    if (are_different("initial_learning_rate_min", initial_learning_rate_min, other->initial_learning_rate_min)) return false;
    if (are_different("initial_learning_rate_max", initial_learning_rate_max, other->initial_learning_rate_max)) return false;
    if (are_different("learning_rate_min", learning_rate_min, other->learning_rate_min)) return false;
    if (are_different("learning_rate_max", learning_rate_max, other->learning_rate_max)) return false;

    if (are_different("initial_learning_rate_delta_min", initial_learning_rate_delta_min, other->initial_learning_rate_delta_min)) return false;
    if (are_different("initial_learning_rate_delta_max", initial_learning_rate_delta_max, other->initial_learning_rate_delta_max)) return false;
    if (are_different("learning_rate_delta_min", learning_rate_delta_min, other->learning_rate_delta_min)) return false;
    if (are_different("learning_rate_delta_max", learning_rate_delta_max, other->learning_rate_delta_max)) return false;

    if (are_different("initial_weight_decay_min", initial_weight_decay_min, other->initial_weight_decay_min)) return false;
    if (are_different("initial_weight_decay_max", initial_weight_decay_max, other->initial_weight_decay_max)) return false;
    if (are_different("weight_decay_min", weight_decay_min, other->weight_decay_min)) return false;
    if (are_different("weight_decay_max", weight_decay_max, other->weight_decay_max)) return false;

    if (are_different("initial_weight_decay_delta_min", initial_weight_decay_delta_min, other->initial_weight_decay_delta_min)) return false;
    if (are_different("initial_weight_decay_delta_max", initial_weight_decay_delta_max, other->initial_weight_decay_delta_max)) return false;
    if (are_different("weight_decay_delta_min", weight_decay_delta_min, other->weight_decay_delta_min)) return false;
    if (are_different("weight_decay_delta_max", weight_decay_delta_max, other->weight_decay_delta_max)) return false;

    if (are_different("epsilon", epsilon, other->epsilon)) return false;

    if (are_different("initial_alpha_min", initial_alpha_min, other->initial_alpha_min)) return false;
    if (are_different("initial_alpha_max", initial_alpha_max, other->initial_alpha_max)) return false;
    if (are_different("alpha_min", alpha_min, other->alpha_min)) return false;
    if (are_different("alpha_max", alpha_max, other->alpha_max)) return false;

    if (are_different("initial_velocity_reset_min", initial_velocity_reset_min, other->initial_velocity_reset_min)) return false;
    if (are_different("initial_velocity_reset_max", initial_velocity_reset_max, other->initial_velocity_reset_max)) return false;
    if (are_different("velocity_reset_min", velocity_reset_min, other->velocity_reset_min)) return false;
    if (are_different("velocity_reset_max", velocity_reset_max, other->velocity_reset_max)) return false;

    if (are_different("initial_input_dropout_probability_min", initial_input_dropout_probability_min, other->initial_input_dropout_probability_min)) return false;
    if (are_different("initial_input_dropout_probability_max", initial_input_dropout_probability_max, other->initial_input_dropout_probability_max)) return false;
    if (are_different("input_dropout_probability_min", input_dropout_probability_min, other->input_dropout_probability_min)) return false;
    if (are_different("input_dropout_probability_max", input_dropout_probability_max, other->input_dropout_probability_max)) return false;

    if (are_different("initial_hidden_dropout_probability_min", initial_hidden_dropout_probability_min, other->initial_hidden_dropout_probability_min)) return false;
    if (are_different("initial_hidden_dropout_probability_max", initial_hidden_dropout_probability_max, other->initial_hidden_dropout_probability_max)) return false;
    if (are_different("hidden_dropout_probability_min", hidden_dropout_probability_min, other->hidden_dropout_probability_min)) return false;
    if (are_different("hidden_dropout_probability_max", hidden_dropout_probability_max, other->hidden_dropout_probability_max)) return false;

    if (are_different("reset_weights_chance", reset_weights_chance, other->reset_weights_chance)) return false;

    if (are_different("crossover_rate", crossover_rate, other->crossover_rate)) return false;
    if (are_different("more_fit_parent_crossover", more_fit_parent_crossover, other->more_fit_parent_crossover)) return false;
    if (are_different("less_fit_parent_crossover", less_fit_parent_crossover, other->less_fit_parent_crossover)) return false;
    if (are_different("crossover_alter_edge_type", crossover_alter_edge_type, other->crossover_alter_edge_type)) return false;

    if (are_different("number_mutations", number_mutations, other->number_mutations)) return false;

    if (are_different("edge_alter_type", edge_alter_type, other->edge_alter_type)) return false;
    if (are_different("edge_disable", edge_disable, other->edge_disable)) return false;
    if (are_different("edge_enable", edge_enable, other->edge_enable)) return false;
    if (are_different("edge_split", edge_split, other->edge_split)) return false;
    if (are_different("edge_add", edge_add, other->edge_add)) return false;
    if (are_different("node_change_size", node_change_size, other->node_change_size)) return false;
    if (are_different("node_change_size_x", node_change_size_x, other->node_change_size_x)) return false;
    if (are_different("node_change_size_y", node_change_size_y, other->node_change_size_y)) return false;
    if (are_different("node_add", node_add, other->node_add)) return false;
    if (are_different("node_split", node_split, other->node_split)) return false;
    if (are_different("node_merge", node_merge, other->node_merge)) return false;
    if (are_different("node_enable", node_enable, other->node_enable)) return false;
    if (are_different("node_disable", node_disable, other->node_disable)) return false;

    if (are_different("inserted_from_map", inserted_from_map, other->inserted_from_map)) return false;
    if (are_different("generated_from_map", generated_from_map, other->generated_from_map)) return false;

    //genomes
    for (uint32_t i = 0; i < genomes.size(); i++) {
        if (!genomes[i]->is_identical(other->genomes[i], testing_checkpoint)) {
            cerr << "IDENTICAL ERROR: genomes[" << i << "] are not the same!" << endl;
            return false;
        }
    }

    return true;
}
