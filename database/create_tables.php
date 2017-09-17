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
    `training_filename` varchar(256) NOT NULL,
    `validation_filename` varchar(256) NOT NULL,
    `test_filename` varchar(256) NOT NULL,

    `number_training_images` int(11) NOT NULL,
    `number_validation_images` int(11) NOT NULL,
    `number_test_images` int(11) NOT NULL,

    `padding` int(11) NOT NULL,

    `image_channels` int(11) NOT NULL,
    `image_rows` int(11) NOT NULL,
    `image_cols` int(11) NOT NULL,
    `number_classes` int(11) NOT NULL,

    `population_size` int(11) NOT NULL,
    `node_innovation_count` int(11) NOT NULL,
    `edge_innovation_count` int(11) NOT NULL,

    `best_predictions_genome_id` int(11) NOT NULL,

    `genomes_generated` int(11) NOT NULL,
    `inserted_genomes` int(11) NOT NULL,
    `max_genomes` int(11) NOT NULL,

    `reset_weights` tinyint NOT NULL,
    `max_epochs` int(11) NOT NULL,

    `initial_batch_size_min` int(11) NOT NULL,
    `initial_batch_size_max` int(11) NOT NULL,
    `batch_size_min` int(11) NOT NULL,
    `batch_size_max` int(11) NOT NULL,

    `initial_mu_min` float NOT NULL,
    `initial_mu_max` float NOT NULL,
    `mu_min` float NOT NULL,
    `mu_max` float NOT NULL,

    `initial_mu_delta_min` float NOT NULL,
    `initial_mu_delta_max` float NOT NULL,
    `mu_delta_min` float NOT NULL,
    `mu_delta_max` float NOT NULL,

    `initial_learning_rate_min` float NOT NULL,
    `initial_learning_rate_max` float NOT NULL,
    `learning_rate_min` float NOT NULL,
    `learning_rate_max` float NOT NULL,

    `initial_learning_rate_delta_min` float NOT NULL,
    `initial_learning_rate_delta_max` float NOT NULL,
    `learning_rate_delta_min` float NOT NULL,
    `learning_rate_delta_max` float NOT NULL,

    `initial_weight_decay_min` float NOT NULL,
    `initial_weight_decay_max` float NOT NULL,
    `weight_decay_min` float NOT NULL,
    `weight_decay_max` float NOT NULL,

    `initial_weight_decay_delta_min` float NOT NULL,
    `initial_weight_decay_delta_max` float NOT NULL,
    `weight_decay_delta_min` float NOT NULL,
    `weight_decay_delta_max` float NOT NULL,

    `epsilon` float NOT NULL,

    `initial_alpha_min` float NOT NULL,
    `initial_alpha_max` float NOT NULL,
    `alpha_min` float NOT NULL,
    `alpha_max` float NOT NULL,

    `initial_velocity_reset_min` int(11) NOT NULL,
    `initial_velocity_reset_max` int(11) NOT NULL,
    `velocity_reset_min` int(11) NOT NULL,
    `velocity_reset_max` int(11) NOT NULL,

    `initial_input_dropout_probability_min` float NOT NULL,
    `initial_input_dropout_probability_max` float NOT NULL,
    `input_dropout_probability_min` float NOT NULL,
    `input_dropout_probability_max` float NOT NULL,

    `initial_hidden_dropout_probability_min` float NOT NULL,
    `initial_hidden_dropout_probability_max` float NOT NULL,
    `hidden_dropout_probability_min` float NOT NULL,
    `hidden_dropout_probability_max` float NOT NULL,

    `reset_weights_chance` float NOT NULL,

    `crossover_rate` float NOT NULL,
    `more_fit_parent_crossover` float NOT NULL,
    `less_fit_parent_crossover` float NOT NULL,
    `crossover_alter_edge_type` float NOT NULL,

    `number_mutations` int(11) NOT NULL,
    `edge_alter_type` float NOT NULL,
    `edge_disable` float NOT NULL,
    `edge_enable` float NOT NULL,
    `edge_split` float NOT NULL,
    `edge_add` float NOT NULL,
    `node_change_size` float NOT NULL,
    `node_change_size_x` float NOT NULL,
    `node_change_size_y` float NOT NULL,
    `node_add` float NOT NULL,
    `node_split` float NOT NULL,
    `node_merge` float NOT NULL,
    `node_enable` float NOT NULL,
    `node_disable` float NOT NULL,

    `generator` varchar(64) NOT NULL,
    `normal_distribution` varchar(128) NOT NULL,
    `rng_long` varchar(64) NOT NULL,
    `rng_float` varchar(64) NOT NULL,

    `inserted_from_map` varchar(256) NOT NULL,
    `generated_from_map` varchar(256) NOT NULL,

    PRIMARY KEY(`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1";

query_exact_db($query);

  //all_nodes and all_edges and genomes will reference the ID of this search

$query = "DROP TABLE cnn_genome;";
query_exact_db($query, false);

$query = "CREATE TABLE `cnn_genome` (
    `id` int(11) NOT NULL AUTO_INCREMENT,
    `exact_id` int(11) NOT NULL,

    `input_node_innovation_numbers` BLOB NOT NULL,
    `softmax_node_innovation_numbers` BLOB NOT NULL,

    `generator` varchar(64) NOT NULL,
    `normal_distribution` varchar(128) NOT NULL,

    `hyperparameters` VARCHAR(256) NOT NULL,

    `velocity_reset` int(11) NOT NULL,
    `batch_size` int(11) NOT NULL,

    `epoch` int(11) NOT NULL,
    `max_epochs` int(11) NOT NULL,
    `reset_weights` tinyint(1) NOT NULL,

    `padding` int(11) DEFAULT NULL,

    `best_epoch` int(11) NOT NULL,
    `number_validation_images` int(11) DEFAULT NULL,
    `best_validation_error` float NOT NULL,
    `best_validation_predictions` int(11) NOT NULL,

    `number_training_images` int(11) DEFAULT NULL,
    `training_error` float NOT NULL,
    `training_predictions` int NOT NULL,

    `number_test_images` int(11) DEFAULT NULL,
    `test_error` float DEFAULT NULL,
    `test_predictions` int(11) DEFAULT NULL,

    `started_from_checkpoint` tinyint(1) NOT NULL,

    `generation_id` int(11) NOT NULL,
    `name` varchar(64),
    `checkpoint_filename` varchar(128) NOT NULL,
    `output_filename` varchar(128) NOT NULL,

    `generated_by_map` varchar(256) NOT NULL,

    `stderr_out` blob,

  PRIMARY KEY(`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1";
query_exact_db($query);

$query = "DROP TABLE cnn_edge;";
query_exact_db($query, false);

$query = "CREATE TABLE `cnn_edge` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `exact_id` int(11) NOT NULL,
  `genome_id` int(11) NOT NULL,

  `type` int(11) NOT NULL,
  `innovation_number` int(11) NOT NULL,

  `input_node_innovation_number` int(11) NOT NULL,
  `output_node_innovation_number` int(11) NOT NULL,

  `batch_size` int(11) NOT NULL,
  `filter_x` int(11) NOT NULL,
  `filter_y` int(11) NOT NULL,
  `weights` BLOB NOT NULL,
  `best_weights` BLOB NOT NULL,

  `fixed` tinyint(1) NOT NULL,
  `disabled` tinyint(1) NOT NULL,
  `forward_visited` tinyint(1) NOT NULL,
  `reverse_visited` tinyint(1) NOT NULL,

  `reverse_filter_x` tinyint(1) NOT NULL,
  `reverse_filter_y` tinyint(1) NOT NULL,
  `needs_initialization` tinyint(1) NOT NULL,
  `scale_values` BLOB NOT NULL,

  PRIMARY KEY(`id`),
  UNIQUE KEY(`id`, `exact_id`, `genome_id`, `innovation_number`),
  KEY(`exact_id`, `genome_id`),
  KEY(`exact_id`),
  KEY(`genome_id`),
  KEY(`innovation_number`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1";
query_exact_db($query);

$query = "DROP TABLE cnn_node;";
query_exact_db($query, false);

$query = "CREATE TABLE `cnn_node` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `exact_id` int(11) NOT NULL,
  `genome_id` int(11) NOT NULL,

  `innovation_number` int(11) NOT NULL,
  `depth` float NOT NULL,

  `batch_size` int(11) NOT NULL,
  `size_x` int(11) NOT NULL,
  `size_y` int(11) NOT NULL,

  `type` int(11) NOT NULL,

  `forward_visited` tinyint(1) NOT NULL,
  `reverse_visited` tinyint(1) NOT NULL,
  `weight_count` int(11) NOT NULL,
  `needs_initialization` tinyint(1) NOT NULL,

  `disabled` tinyint(1) NOT NULL,

  `batch_norm_parameters` VARCHAR(256) NOT NULL,

  PRIMARY KEY(`id`),
  UNIQUE KEY(`id`, `exact_id`, `genome_id`, `innovation_number`),
  KEY(`exact_id`, `genome_id`),
  KEY(`exact_id`),
  KEY(`genome_id`),
  KEY(`innovation_number`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1";
query_exact_db($query);




?>
