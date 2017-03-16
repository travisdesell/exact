<?php

$cwd[__FILE__] = __FILE__;
if (is_link($cwd[__FILE__])) $cwd[__FILE__] = readlink($cwd[__FILE__]);
$cwd[__FILE__] = dirname($cwd[__FILE__]);

require_once($cwd[__FILE__] . "/../../citizen_science_grid/my_query.php");

function print_stats($exact_id) {
    $avg_num_weights = 0;

    $genome_result = query_boinc_db("SELECT id FROM cnn_genome WHERE exact_id = $exact_id ORDER BY best_error LIMIT 20");
    while ($genome_row = $genome_result->fetch_assoc()) {
        $id = $genome_row['id'];

        $weight_result = query_boinc_db("SELECT sum(filter_x * filter_y) FROM cnn_edge WHERE genome_id = $id AND disabled = 0");
        $weight_row = $weight_result->fetch_assoc();
        $weight_count = $weight_row['sum(filter_x * filter_y)'];
        $avg_num_weights += $weight_count;

        echo "$id, weights: $weight_count\n";
    }

    $avg_num_weights /= 20;

    $genome_result = query_boinc_db("SELECT min(best_error), avg(best_error), max(best_error), min(test_error), avg(test_error), max(test_error), min(best_predictions), avg(best_predictions), max(best_predictions), min(test_predictions), avg(test_predictions), max(test_predictions) FROM cnn_genome WHERE exact_id = $exact_id ORDER BY best_error LIMIT 20");

    while ($genome_row = $genome_result->fetch_assoc()) {
        $min_error = $genome_row['min(best_error)'];
        $avg_error = $genome_row['avg(best_error)'];
        $max_error = $genome_row['max(best_error)'];
        $min_predictions = $genome_row['min(best_predictions)'] / 600.0;
        $avg_predictions = $genome_row['avg(best_predictions)'] / 600.0;
        $max_predictions = $genome_row['max(best_predictions)'] / 600.0;

        $min_test_error = $genome_row['min(test_error)'];
        $avg_test_error = $genome_row['avg(test_error)'];
        $max_test_error = $genome_row['max(test_error)'];
        $min_test_predictions = $genome_row['min(test_predictions)'] / 100.0;
        $avg_test_predictions = $genome_row['avg(test_predictions)'] / 100.0;
        $max_test_predictions = $genome_row['max(test_predictions)'] / 100.0;

        $avg_num_weights = number_format($avg_num_weights, 2);

        $min_error = number_format($min_error, 2);
        $avg_error = number_format($avg_error, 2);
        $max_error = number_format($max_error, 2);
        $min_test_error = number_format($min_test_error, 2);
        $avg_test_error = number_format($avg_test_error, 2);
        $max_test_error = number_format($max_test_error, 2);

        $min_predictions = number_format($min_predictions, 2);
        $avg_predictions = number_format($avg_predictions, 2);
        $max_predictions = number_format($max_predictions, 2);
        $min_test_predictions = number_format($min_test_predictions, 2);
        $avg_test_predictions = number_format($avg_test_predictions, 2);
        $max_test_predictions = number_format($max_test_predictions, 2);

        echo "$avg_num_weights & $min_error & $avg_error & $max_error & $min_test_error & $avg_test_error & $max_test_error & $max_predictions & $avg_predictions & $min_predictions & $max_test_predictions & $avg_test_predictions & $min_test_predictions \n";
    }
}

print_stats(20);
print_stats(21);

?>
