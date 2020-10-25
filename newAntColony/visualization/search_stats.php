<?php

$cwd[__FILE__] = __FILE__;
if (is_link($cwd[__FILE__])) $cwd[__FILE__] = readlink($cwd[__FILE__]);
$cwd[__FILE__] = dirname($cwd[__FILE__]);

require_once($cwd[__FILE__] . "/../www/my_query.php");


function get_num_edges($db_name, $exact_id, $type, &$best_edges, &$worst_edges, &$best_weights, &$worst_weights) {
    $query = "SELECT id FROM cnn_genome WHERE exact_id = $exact_id ORDER BY $type";
    echo $query . "\n";
    $genome_result = query_multi_db($db_name, $query);

    $genome_row = $genome_result->fetch_assoc();
    $id = $genome_row['id'];

    $weight_result = query_multi_db($db_name, "SELECT sum(filter_x * filter_y) FROM cnn_edge WHERE genome_id = $id AND disabled = 0 AND forward_visited = 1 AND reverse_visited = 1");
    $weight_row = $weight_result->fetch_assoc();
    $best_weights = $weight_row['sum(filter_x * filter_y)'];
 
    $edge_result = query_multi_db($db_name, "SELECT count(id) FROM cnn_edge WHERE genome_id = $id AND disabled = 0 AND forward_visited = 1 AND reverse_visited = 1");
    $edge_row = $edge_result->fetch_assoc();
    $best_edges = $edge_row['count(id)'];
 
    $genome_result = query_multi_db($db_name, "SELECT id FROM cnn_genome WHERE exact_id = $exact_id ORDER BY $type DESC");
    $genome_row = $genome_result->fetch_assoc();
    $id = $genome_row['id'];

    $weight_result = query_multi_db($db_name, "SELECT sum(filter_x * filter_y) FROM cnn_edge WHERE genome_id = $id AND disabled = 0 AND forward_visited = 1 AND reverse_visited = 1");
    $weight_row = $weight_result->fetch_assoc();
    $worst_weights = $weight_row['sum(filter_x * filter_y)'];
 
    $edge_result = query_multi_db($db_name, "SELECT count(id) FROM cnn_edge WHERE genome_id = $id AND disabled = 0 AND forward_visited = 1 AND reverse_visited = 1");
    $edge_row = $edge_result->fetch_assoc();
    $worst_edges = $edge_row['count(id)'];
}

function add_stats($db_name, $exact_id, &$stats) {
    $avg_num_weights = 0;
    $avg_num_edges = 0;

    $genome_result = query_multi_db($db_name, "SELECT id FROM cnn_genome WHERE exact_id = $exact_id ORDER BY best_error");
    $count = 0;
    while ($genome_row = $genome_result->fetch_assoc()) {
        $id = $genome_row['id'];

        $weight_result = query_multi_db($db_name, "SELECT sum(filter_x * filter_y) FROM cnn_edge WHERE genome_id = $id AND disabled = 0 AND forward_visited = 1 AND reverse_visited = 1");
        $weight_row = $weight_result->fetch_assoc();
        $weight_count = $weight_row['sum(filter_x * filter_y)'];
        $avg_num_weights += $weight_count;


        $edge_result = query_multi_db($db_name, "SELECT count(id) FROM cnn_edge WHERE genome_id = $id AND disabled = 0 AND forward_visited = 1 AND reverse_visited = 1");
        $edge_row = $edge_result->fetch_assoc();
        $edge_count = $edge_row['count(id)'];
        $avg_num_edges += $edge_count;

        $count++;
    }

    $best_training_edges = 0;
    $worst_training_edges = 0;
    $best_training_weights = 0;
    $worst_training_weights = 0;

    $best_generalizability_edges = 0;
    $worst_generalizability_edges = 0;
    $best_generalizability_weights = 0;
    $worst_generalizability_weights = 0;

    $best_test_edges = 0;
    $worst_test_edges = 0;
    $best_test_weights = 0;
    $worst_test_weights = 0;

    $best_gen_test_edges = 0;
    $worst_gen_test_edges = 0;
    $best_gen_test_weights = 0;
    $worst_gen_test_weights = 0;

    get_num_edges($db_name, $exact_id, "best_error", $best_training_edges, $worst_training_edges, $best_training_weights, $worst_training_weights);
    get_num_edges($db_name, $exact_id, "generalizability_error", $best_generalizability_edges, $worst_generalizability_edges, $best_generalizability_weights, $worst_generalizability_weights);
    get_num_edges($db_name, $exact_id, "test_error", $best_test_edges, $worst_test_edges, $best_test_weights, $worst_test_weights);
    get_num_edges($db_name, $exact_id, "(test_error + generalizability_error)", $best_gen_test_edges, $worst_gen_test_edges, $best_gen_test_weights, $worst_gen_test_weights);


    echo "$id, weights: $weight_count\n";
    echo "best_training_edges: $best_training_edges, worst_training_edges: $worst_training_edges\n";
    echo "best_training_weights: $best_training_weights, worst_training_weights: $worst_training_weights\n";

    echo "best_generalizability_edges: $best_generalizability_edges, worst_generalizability_edges: $worst_generalizability_edges\n";
    echo "best_generalizability_weights: $best_generalizability_weights, worst_generalizability_weights: $worst_generalizability_weights\n";

    echo "best_test_edges: $best_test_edges, worst_test_edges: $worst_test_edges\n";
    echo "best_test_weights: $best_test_weights, worst_test_weights: $worst_test_weights\n";

    echo "best_gen_test_edges: $best_gen_test_edges, worst_gen_test_edges: $worst_gen_test_edges\n";
    echo "best_gen_test_weights: $best_gen_test_weights, worst_gen_test_weights: $worst_gen_test_weights\n";


    $avg_num_weights /= $count;
    $avg_num_edges /= $count;

    $stat = array();
    $stat['avg_num_weights'] = number_format($avg_num_weights, 3);
    $stat['avg_num_edges'] = number_format($avg_num_edges, 3);

    $stat['best_training_weights'] = $best_training_weights;
    $stat['worst_training_weights'] = $worst_training_weights;

    $stat['best_generalizability_weights'] = $best_generalizability_weights;
    $stat['worst_generalizability_weights'] = $worst_generalizability_weights;

    $stat['best_test_weights'] = $best_test_weights;
    $stat['worst_test_weights'] = $worst_test_weights;

    $stat['best_gen_test_weights'] = $best_gen_test_weights;
    $stat['worst_gen_test_weights'] = $worst_gen_test_weights;


    $stat['best_training_edges'] = $best_training_edges;
    $stat['worst_training_edges'] = $worst_training_edges;

    $stat['best_generalizability_edges'] = $best_generalizability_edges;
    $stat['worst_generalizability_edges'] = $worst_generalizability_edges;

    $stat['best_test_edges'] = $best_test_edges;
    $stat['worst_test_edges'] = $worst_test_edges;

    $stat['best_gen_test_edges'] = $best_gen_test_edges;
    $stat['worst_gen_test_edges'] = $worst_gen_test_edges;

    $query = "SELECT "
            . "min(best_error), avg(best_error), max(best_error), "
            . "min(generalizability_error), avg(generalizability_error), max(generalizability_error), "
            . "min(test_error), avg(test_error), max(test_error), "
            . "min(generalizability_error + test_error), avg(generalizability_error + test_error), max(generalizability_error + test_error), "
            . "min(best_predictions), avg(best_predictions), max(best_predictions), "
            . "min(generalizability_predictions), avg(generalizability_predictions), max(generalizability_predictions), "
            . "min(test_predictions), avg(test_predictions), max(test_predictions), "
            . "min(generalizability_predictions + test_predictions), avg(generalizability_predictions + test_predictions), max(generalizability_predictions + test_predictions) "
            . "FROM cnn_genome WHERE exact_id = $exact_id ORDER BY best_error";

    //echo $query . "\n";
    $genome_result = query_multi_db($db_name, $query);

    while ($genome_row = $genome_result->fetch_assoc()) {
        $stat['best_training_error'] = number_format($genome_row['min(best_error)'], 3);
        $stat['avg_training_error'] = number_format($genome_row['avg(best_error)'], 3);
        $stat['worst_training_error'] = number_format($genome_row['max(best_error)'], 3);

        $stat['best_generalizability_error'] = number_format($genome_row['min(generalizability_error)'], 3);
        $stat['avg_generalizability_error'] = number_format($genome_row['avg(generalizability_error)'], 3);
        $stat['worst_generalizability_error'] = number_format($genome_row['max(generalizability_error)'], 3);

        $stat['best_test_error'] = number_format($genome_row['min(test_error)'], 3);
        $stat['avg_test_error'] = number_format($genome_row['avg(test_error)'], 3);
        $stat['worst_test_error'] = number_format($genome_row['max(test_error)'], 3);

        $stat['best_gen_test_error'] = number_format($genome_row['min(generalizability_error + test_error)'], 3);
        $stat['avg_gen_test_error'] = number_format($genome_row['avg(generalizability_error + test_error)'], 3);
        $stat['worst_gen_test_error'] = number_format($genome_row['max(generalizability_error + test_error)'], 3);


        $stat['best_training_predictions'] = number_format($genome_row['max(best_predictions)'] / 600, 3);
        $stat['avg_training_predictions'] = number_format($genome_row['avg(best_predictions)'] / 600, 3);
        $stat['worst_training_predictions'] = number_format($genome_row['min(best_predictions)'] / 600, 3);

        $stat['best_generalizability_predictions'] = number_format($genome_row['max(generalizability_predictions)'] / 50, 3);
        $stat['avg_generalizability_predictions'] = number_format($genome_row['avg(generalizability_predictions)'] / 50, 3);
        $stat['worst_generalizability_predictions'] = number_format($genome_row['min(generalizability_predictions)'] / 50, 3);

        $stat['best_test_predictions'] = number_format($genome_row['max(test_predictions)'] / 50, 3);
        $stat['avg_test_predictions'] = number_format($genome_row['avg(test_predictions)'] / 50, 3);
        $stat['worst_test_predictions'] = number_format($genome_row['min(test_predictions)'] / 50, 3);

        $stat['best_gen_test_predictions'] = number_format($genome_row['max(generalizability_predictions + test_predictions)'] / 100, 3);
        $stat['avg_gen_test_predictions'] = number_format($genome_row['avg(generalizability_predictions + test_predictions)'] / 100, 3);
        $stat['worst_gen_test_predictions'] = number_format($genome_row['min(generalizability_predictions + test_predictions)'] / 100, 3);
    }

    //print var_dump($stat);

    $stats[] = $stat;
}

$stats = array();

for ($i = 24; $i < 34; $i++) {
    add_stats("exact_batchnorm", $i, $stats);
}

function avg_stats($min, $max, &$stats) {
    $stat = array();

    $keys = array_keys($stats[0]);
    print "keys: " . var_dump($keys) . "\n";

    foreach ($keys as $key) {
        echo "averaging stats for $key, min: $min, max: $max, \n";

        $value = 0;

        for ($i = $min; $i < $max; $i++) {
            $value += floatval(str_replace( ',', '', $stats[$i][$key]));

            echo "adding to value: " . $stats[$i][$key] . ", now: $value\n";
        }

        echo "value before divide: $value dividing by: " . ($max - $min) . "\n";

        $value /= ($max - $min);
        $stat[$key] = number_format($value, 3);

        echo "setting average '$key' to: $value\n";
    }
    $stats[] = $stat;
}

avg_stats(0, 5, $stats);

avg_stats(5, 10, $stats);

function bold_best($min, $max, &$stats) {
    foreach (array_keys($stats[$min]) as $key) {
        echo "bolding $key from $min to $max\n";
        $best_index = $min;
        $best_value = $stats[$min][$key];

        if (strpos($key, "pred") !== false) {
            for ($i = $min + 1; $i < $max; $i++) {
                echo "predictions checking " . $stats[$i][$key];
                if (floatval($stats[$i][$key]) > floatval($best_value)) {
                    echo " -- new best!";
                    $best_value = $stats[$i][$key];
                    $best_index = $i;
                }
                echo "\n";
            }
        } else {
            for ($i = $min + 1; $i < $max; $i++) {
                echo "not predictions checking " . $stats[$i][$key];
                if ($stats[$i][$key] < $best_value) {
                    echo " -- new best!";
                    $best_value = $stats[$i][$key];
                    $best_index = $i;
                }
                echo "\n";
            }
         }

        $stats[$best_index][$key] = "\\textbf{" . $stats[$best_index][$key] . "}";
    }
}

bold_best(0, 10, $stats);
bold_best(10, 12, $stats);

function print_row($name, $col_name, $stats) {
    echo "\\textbf{" . $name . "} ";
    $count = 0;
    foreach ($stats as $stat) {
        if ($count == 5) echo "& ";

        echo "& " . $stat[$col_name];
        $count++;

        if ($count == 10) echo "& ";
    }
    echo "\\\\\n";
    echo "\hline\n";
}

echo "
\\begin{table*}
\\begin{tiny}
\\begin{center}
\\begin{tabular}{|l|r|r|r|r|r|c|r|r|r|r|r|c|r|r|}
\\hline
    &  \\textbf{Simplex 1}  &  \\textbf{Simplex 2}  &  \\textbf{Simplex 3}  &  \\textbf{Simplex 4}  &  \\textbf{Simplex 5}  & &  \\textbf{Fixed 1}  &  \\textbf{Fixed 2}  &  \\textbf{Fixed 3}  &  \\textbf{Fixed 4}  &  \\textbf{Fixed 5}  &  &  \\textbf{Avg Simplex}  &  \\textbf{Avg Fixed} \\\\
\\hline
\\hline
";

print_row("Average", "avg_num_weights", $stats);
echo "\hline\n";

print_row("Best Training Error", "best_training_weights", $stats);
print_row("Best Gen Error", "best_generalizability_weights", $stats);
print_row("Best Test Error", "best_test_weights", $stats);
print_row("Best Gen+Test Error", "best_gen_test_weights", $stats);

echo "\hline\n";

print_row("Worst Training Error", "worst_training_weights", $stats);
print_row("Worst Test Error", "worst_test_weights", $stats);
print_row("Worst Gen Error", "worst_generalizability_weights", $stats);
print_row("Worst Gen+Test Error", "worst_gen_test_weights", $stats);

echo "\\end{tabular}
\\caption{\\label{table:benchmark_nns} Evolved Neural Network Weight Counts}
\\end{center}
\\end{tiny}
\\end{table*}
";


echo "
\\begin{table*}
\\begin{tiny}
\\begin{center}
\\begin{tabular}{|l|r|r|r|r|r|c|r|r|r|r|r|c|r|r|}
\\hline
    &  \\textbf{Simplex 1}  &  \\textbf{Simplex 2}  &  \\textbf{Simplex 3}  &  \\textbf{Simplex 4}  &  \\textbf{Simplex 5}  & &  \\textbf{Fixed 1}  &  \\textbf{Fixed 2}  &  \\textbf{Fixed 3}  &  \\textbf{Fixed 4}  &  \\textbf{Fixed 5}  &  &  \\textbf{Avg Simplex}  &  \\textbf{Avg Fixed} \\\\
\\hline
\\hline
";

print_row("Average", "avg_num_edges", $stats);

echo "\hline\n";

print_row("Best Training Error", "best_training_edges", $stats);
print_row("Best Gen Error", "best_generalizability_edges", $stats);
print_row("Best Test Error", "best_test_edges", $stats);
print_row("Best Gen+Test Error", "best_gen_test_edges", $stats);

echo "\hline\n";

print_row("Worst Training Error", "worst_training_edges", $stats);
print_row("Worst Gen Error", "worst_generalizability_edges", $stats);
print_row("Worst Test Error", "worst_test_edges", $stats);
print_row("Worst Gen+Test Error", "worst_gen_test_edges", $stats);

echo "\\end{tabular}
\\caption{\\label{table:benchmark_nns} Evolved Neural Network Edge Counts}
\\end{center}
\\end{tiny}
\\end{table*}
";


echo "
\\begin{table*}
\\begin{tiny}
\\begin{center}
\\begin{tabular}{|l|r|r|r|r|r|c|r|r|r|r|r|c|r|r|}
\\hline
    &  \\textbf{Simplex 1}  &  \\textbf{Simplex 2}  &  \\textbf{Simplex 3}  &  \\textbf{Simplex 4}  &  \\textbf{Simplex 5}  & &  \\textbf{Fixed 1}  &  \\textbf{Fixed 2}  &  \\textbf{Fixed 3}  &  \\textbf{Fixed 4}  &  \\textbf{Fixed 5}  &  &  \\textbf{Avg Simplex}  &  \\textbf{Avg Fixed} \\\\
\\hline
\\hline
";

print_row("Best Training Error", "best_training_error", $stats);
print_row("Best  Gen. Error ", "best_generalizability_error", $stats);
print_row("Best  Testing Error ", "best_test_error", $stats);
print_row("Best  Gen+Test Error ", "best_gen_test_error", $stats);

echo "\hline\n";

print_row("Avg.  Training Error", "avg_training_error", $stats);
print_row("Avg.  Gen. Error ", "avg_generalizability_error", $stats);
print_row("Avg.  Testing Error ", "avg_test_error", $stats);
print_row("Avg.  Gen+Test Error ", "avg_gen_test_error", $stats);

echo "\hline\n";

print_row("Worst Training Error ", "worst_training_error", $stats);
print_row("Worst Gen. Error ", "worst_generalizability_error", $stats);
print_row("Worst Testing Error ", "worst_test_error", $stats);
print_row("Worst Gen+Test Error ", "worst_gen_test_error", $stats);


echo "\\end{tabular}
\\caption{\\label{table:benchmark_nns} Evolved Neural Network Error Rates}
\\end{center}
\\end{tiny}
\\end{table*}
";



echo "
\\begin{table*}
\\begin{tiny}
\\begin{center}
\\begin{tabular}{|l|r|r|r|r|r|c|r|r|r|r|r|c|r|r|}
\\hline
    &  \\textbf{Simplex 1}  &  \\textbf{Simplex 2}  &  \\textbf{Simplex 3}  &  \\textbf{Simplex 4}  &  \\textbf{Simplex 5}  & &  \\textbf{Fixed 1}  &  \\textbf{Fixed 2}  &  \\textbf{Fixed 3}  &  \\textbf{Fixed 4}  &  \\textbf{Fixed 5}  &  &  \\textbf{Avg Simplex}  &  \\textbf{Avg Fixed} \\\\
\\hline
\\hline
";


print_row("Best  Training Pred ", "best_training_predictions", $stats);
print_row("Best  Gen. Pred ", "best_generalizability_predictions", $stats);
print_row("Best  Testing Pred ", "best_test_predictions", $stats);
print_row("Best  Gen+Test Pred ", "best_gen_test_predictions", $stats);

echo "\hline\n";

print_row("Avg.  Training Pred ", "avg_training_predictions", $stats);
print_row("Avg.  Gen. Pred ", "avg_generalizability_predictions", $stats);
print_row("Avg.  Testing Pred ", "avg_test_predictions", $stats);
print_row("Avg.  Gen+Test Pred ", "avg_gen_test_predictions", $stats);

echo "\hline\n";

print_row("Worst Training Pred ", "worst_training_predictions", $stats);
print_row("Worst Gen. Pred ", "worst_generalizability_predictions", $stats);
print_row("Worst Testing Pred ", "worst_test_predictions", $stats);
print_row("Worst Gen+Test Pred ", "worst_gen_test_predictions", $stats);


echo "\\end{tabular}
\\caption{\\label{table:benchmark_nns} Evolved Neural Network Prediction Rates}
\\end{center}
\\end{tiny}
\\end{table*}
";

?>
