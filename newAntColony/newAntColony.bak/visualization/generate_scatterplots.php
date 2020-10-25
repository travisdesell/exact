<?php

$cwd[__FILE__] = __FILE__;
if (is_link($cwd[__FILE__])) $cwd[__FILE__] = readlink($cwd[__FILE__]);
$cwd[__FILE__] = dirname($cwd[__FILE__]);

require_once($cwd[__FILE__] . "/../www/my_query.php");


$db_name = "exact_bn_sfmp";
$min_exact_id = 24;
$max_exact_id = 28;

$query = "SELECT (generalizability_error + test_error), initial_mu, mu_delta, initial_learning_rate, learning_rate_delta, initial_weight_decay, weight_decay_delta, alpha, batch_size, velocity_reset, input_dropout_probability, hidden_dropout_probability FROM cnn_genome WHERE exact_id >= $min_exact_id AND exact_id <= $max_exact_id";
$genome_result = query_multi_db($db_name, $query);

echo "#gen_test_error,initial_mu,mu_delta,initial_learning_rate,learning_rate_delta,initial_weight_decay,weight_decay_delta,alpha,batch_size,velocity_reset,input_dropout_probability,hidden_dropout_probability\n";

while ($genome_row = $genome_result->fetch_assoc()) {
        $gen_test_error = $genome_row['(generalizability_error + test_error)'];

        $initial_mu = $genome_row['initial_mu'];
        $mu_delta = $genome_row['mu_delta'];

        $initial_learning_rate = $genome_row['initial_learning_rate'];
        $learning_rate_delta = $genome_row['learning_rate_delta'];

        $initial_weight_decay = $genome_row['initial_weight_decay'];
        $weight_decay_delta = $genome_row['weight_decay_delta'];

        $alpha = $genome_row['alpha'];
        $batch_size = $genome_row['batch_size'];
        $velocity_reset = $genome_row['velocity_reset'];
        $input_dropout_probability = $genome_row['input_dropout_probability'];
        $hidden_dropout_probability = $genome_row['hidden_dropout_probability'];

        echo "$gen_test_error,$initial_mu,$mu_delta,$initial_learning_rate,$learning_rate_delta,$initial_weight_decay,$weight_decay_delta,$alpha,$batch_size,$velocity_reset,$input_dropout_probability,$hidden_dropout_probability\n";
}

?>
