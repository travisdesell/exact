<?php

//first log in as root to create the mysql database:
//
//create user 'exact_user'@'localhost' IDENTIFIED BY 'ev0lv3 all tH3 NETW0rks!!';
//GRANT ALL PRIVILEGES ON exact.* TO 'exact_user'@'localhost';
//flush privileges;

require_once("mysql.php");

$query = "DROP TABLE exact_search;";
query_exact_db($query, false);

$query = "CREATE TABLE `exact_search` (
    `id` int(11) NOT NULL AUTO_INCREMENT,
    `search_name` varchar(128) NOT NULL,
    `output_directory` varchar(128) NOT NULL,

    `number_images` int(11) NOT NULL,
    `image_rows` int(11) NOT NULL,
    `image_cols` int(11) NOT NULL,
    `number_classes` int(11) NOT NULL,

    `population_size` int(11) NOT NULL,
    `node_innovation_count` int(11) NOT NULL,
    `edge_innovation_count` int(11) NOT NULL,

    `genomes_generated` int(11) NOT NULL,
    `inserted_genomes` int(11) NOT NULL,

    `reset_edges` tinyint NOT NULL,
    `min_epochs` int(11) NOT NULL,
    `max_epochs` int(11) NOT NULL,
    `improvement_required_epochs` int(11) NOT NULL,
    `max_individuals` int(11) NOT NULL,

    `learning_rate` double NOT NULL,
    `weight_decay` double NOT NULL,

    `crossover_rate` double NOT NULL,
    `more_fit_parent_crossover` double NOT NULL,
    `less_fit_parent_crossover` double NOT NULL,

    `number_mutations` int(11) NOT NULL,
    `edge_disable` double NOT NULL,
    `edge_enable` double NOT NULL,
    `edge_split` double NOT NULL,
    `edge_add` double NOT NULL,
    `edge_change_stride` double NOT NULL,
    `node_change_size` double NOT NULL,
    `node_change_size_x` double NOT NULL,
    `node_change_size_y` double NOT NULL,
    `node_change_pool_size` double NOT NULL,

    `inserted_from_disable_edge` int NOT NULL,
    `inserted_from_enable_edge` int NOT NULL,
    `inserted_from_split_edge` int NOT NULL,
    `inserted_from_add_edge` int NOT NULL,
    `inserted_from_change_size` int NOT NULL,
    `inserted_from_change_size_x` int NOT NULL,
    `inserted_from_change_size_y` int NOT NULL,
    `inserted_from_crossover` int NOT NULL,

    `generator` varchar(64) NOT NULL,
    `normal_distribution` varchar(128) NOT NULL,
    `rng_long` varchar(64) NOT NULL,
    `rng_double` varchar(64) NOT NULL,

    PRIMARY KEY(`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1";

query_exact_db($query);

  //all_nodes and all_edges and genomes will reference the ID of this search

$query = "DROP TABLE cnn_genome;";
query_exact_db($query, false);

$query = "CREATE TABLE `cnn_genome` (
    `id` int(11) NOT NULL AUTO_INCREMENT,
    `exact_id` int(11) NOT NULL,

    `input_node_innovation_number` int NOT NULL,
    `softmax_node_innovation_numbers` BLOB NOT NULL,

    `generator` varchar(64) NOT NULL,
    `normal_distribution` varchar(128) NOT NULL,

    `initial_mu` double NOT NULL,
    `mu` double NOT NULL,
    `initial_learning_rate` double NOT NULL,
    `learning_rate` double NOT NULL,
    `initial_weight_decay` double NOT NULL,
    `weight_decay` double NOT NULL,

    `epoch` int(11) NOT NULL,
    `min_epochs` int(11) NOT NULL,
    `max_epochs` int(11) NOT NULL,
    `improvement_required_epochs` int(11) NOT NULL,
    `reset_edges` tinyint(1) NOT NULL,

    `best_error` double NOT NULL,
    `best_error_epoch` int(11) NOT NULL,
    `best_predictions` int(11) NOT NULL,
    `best_predictions_epoch` int(11) NOT NULL,

    `best_class_error` BLOB NOT NULL,
    `best_correct_predictions` BLOB NOT NULL,

    `started_from_checkpoint` tinyint(1) NOT NULL,
    `backprop_order` LONGBLOB NOT NULL,

    `generation_id` int(11) NOT NULL,
    `name` varchar(64),
    `checkpoint_filename` varchar(128) NOT NULL,
    `output_filename` varchar(128) NOT NULL,

    `generated_by_disable_edge` int(11) NOT NULL,
    `generated_by_enable_edge` int(11) NOT NULL,
    `generated_by_split_edge` int(11) NOT NULL,
    `generated_by_add_edge` int(11) NOT NULL,
    `generated_by_change_size` int(11) NOT NULL,
    `generated_by_change_size_x` int(11) NOT NULL,
    `generated_by_change_size_y` int(11) NOT NULL,
    `generated_by_crossover` int(11) NOT NULL,

  PRIMARY KEY(`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1";
query_exact_db($query);

$query = "DROP TABLE cnn_edge;";
query_exact_db($query, false);

$query = "CREATE TABLE `cnn_edge` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `exact_id` int(11) NOT NULL,
  `genome_id` int(11) NOT NULL,

  `innovation_number` int(11) NOT NULL,

  `input_node_innovation_number` int(11) NOT NULL,
  `output_node_innovation_number` int(11) NOT NULL,

  `filter_x` int(11) NOT NULL,
  `filter_y` int(11) NOT NULL,
  `weights` BLOB NOT NULL,
  `best_weights` BLOB NOT NULL,
  `previous_velocity` BLOB NOT NULL,

  `fixed` tinyint(1) NOT NULL,
  `disabled` tinyint(1) NOT NULL,
  `reverse_filter_x` tinyint(1) NOT NULL,
  `reverse_filter_y` tinyint(1) NOT NULL,

  PRIMARY KEY(`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1";
query_exact_db($query);

$query = "DROP TABLE cnn_node;";
query_exact_db($query, false);

$query = "CREATE TABLE `cnn_node` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `exact_id` int(11) NOT NULL,
  `genome_id` int(11) NOT NULL,

  `innovation_number` int(11) NOT NULL,
  `depth` double NOT NULL,

  `size_x` int(11) NOT NULL,
  `size_y` int(11) NOT NULL,

  `values` BLOB NOT NULL,
  `errors` BLOB NOT NULL,
  `bias` BLOB NOT NULL,
  `best_bias` BLOB NOT NULL,
  `bias_velocity` BLOB NOT NULL,

  `type` int(11) NOT NULL,
  `total_inputs` int(11) NOT NULL,
  `inputs_fired` int(11) NOT NULL,

  `visited` tinyint(1) NOT NULL,

  PRIMARY KEY(`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1";
query_exact_db($query);




?>
