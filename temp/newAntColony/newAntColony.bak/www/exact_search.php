<?php

$cwd[__FILE__] = __FILE__;
if (is_link($cwd[__FILE__])) $cwd[__FILE__] = readlink($cwd[__FILE__]);
$cwd[__FILE__] = dirname($cwd[__FILE__]);

require_once($cwd[__FILE__] . "/../../citizen_science_grid/header.php");
require_once($cwd[__FILE__] . "/../../citizen_science_grid/navbar.php");
require_once($cwd[__FILE__] . "/../../citizen_science_grid/footer.php");
require_once($cwd[__FILE__] . "/my_query.php");

$search_id = $boinc_db->real_escape_string($_GET['id']);
$db_name = $boinc_db->real_escape_string($_GET['db']);

$exact_result = query_multi_db($db_name, "SELECT search_name FROM exact_search WHERE id = $search_id");
$exact_row = $exact_result->fetch_assoc();
$search_name = $exact_row['search_name'];


print_header("EXACT: $search_name", "<script type='text/javascript' src=''></script>", "wildlife");
print_navbar("Projects: Wildlife@Home", "Wildlife@Home", "..");

echo "
    <div class='container'>";

echo "<h3>$search_name</h3>";

echo "
        <div class='row' style='margin-bottom:10px;'>
            <div class='col-sm-12'>";

echo "<div class='col-sm-4'> <img src='./progress/" . $search_name . "_fitness_progress.png' width='100%'></img> </div>";
echo "<div class='col-sm-4'> <img src='./progress/" . $search_name . "_epochs_progress.png' width='100%'></img> </div>";

if (file_exists('./progress/' . $search_name . '_generated_progress.png')) {
    echo "<div class='col-sm-4'> <img src='./progress/" . $search_name . "_generated_progress.png' width='100%'></img> </div>";
} else {
    echo "<div class='col-sm-4'> <img src='./progress/" . $search_name . "_nodes_edges_progress.png' width='100%'></img> </div>";

    echo "
                </div> <!--col-sm-12 -->
                <div class='col-sm-12'>";

    echo "<div class='col-sm-4'> <img src='./progress/" . $search_name . "_node_percentage_progress.png' width='100%'></img> </div>";
    echo "<div class='col-sm-4'> <img src='./progress/" . $search_name . "_edge_percentage_progress.png' width='100%'></img> </div>";
    echo "<div class='col-sm-4'> <img src='./progress/" . $search_name . "_other_percentage_progress.png' width='100%'></img> </div>";

    echo "<div class='col-sm-4'> <img src='./progress/" . $search_name . "_weights_progress.png' width='100%'></img> </div>";
}

echo "
            </div> <!--col-sm-12 -->
            <div class='col-sm-12'>";

echo "<div class='col-sm-4'> <img src='./hyperparameters/" . $search_name . "_initial_mu.png' width='100%'></img> </div>";
echo "<div class='col-sm-4'> <img src='./hyperparameters/" . $search_name . "_mu_delta.png' width='100%'></img> </div>";
echo "<div class='col-sm-4'> <img src='./hyperparameters/" . $search_name . "_alpha.png' width='100%'></img> </div>";

echo "<div class='col-sm-4'> <img src='./hyperparameters/" . $search_name . "_initial_learning_rate.png' width='100%'></img> </div>";
echo "<div class='col-sm-4'> <img src='./hyperparameters/" . $search_name . "_learning_rate_delta.png' width='100%'></img> </div>";
echo "<div class='col-sm-4'> <img src='./hyperparameters/" . $search_name . "_batch_size.png' width='100%'></img> </div>";

echo "<div class='col-sm-4'> <img src='./hyperparameters/" . $search_name . "_initial_weight_decay.png' width='100%'></img> </div>";
echo "<div class='col-sm-4'> <img src='./hyperparameters/" . $search_name . "_weight_decay_delta.png' width='100%'></img> </div>";
echo "<div class='col-sm-4'> <img src='./hyperparameters/" . $search_name . "_velocity_reset.png' width='100%'></img> </div>";

echo "<div class='col-sm-4'> <img src='./hyperparameters/" . $search_name . "_input_dropout.png' width='100%'></img> </div>";
echo "<div class='col-sm-4'> <img src='./hyperparameters/" . $search_name . "_hidden_dropout.png' width='100%'></img> </div>";

echo "
            </div> <!-- col-sm-12 -->
        </div> <!-- row -->";

$search_rows = array();


if ($db_name == "exact_bn_pool") {
    $best_pred_result = query_multi_db($db_name, "SELECT best_predictions_genome_id FROM exact_search WHERE id = $search_id");
    $best_pred_row = $best_pred_result->fetch_assoc();

    $search_result = query_multi_db($db_name, "SELECT id, best_error, best_predictions, generalizability_error, generalizability_predictions, test_error, test_predictions, best_error_epoch, max_epochs, generation_id, hyperparameters FROM cnn_genome WHERE exact_id = " . $search_id . " AND id = " . $best_pred_row['best_predictions_genome_id']);

    $search_row = $search_result->fetch_assoc();
    $bp = $search_row['best_predictions'];

    if (strpos($search_name, 'cifar') !== false) {
        $search_row['best_error'] = number_format($search_row['best_error'], 3);
        $search_row['best_predictions'] = $bp . " (" . number_format(100.0 * $bp / 50000.00, 2) . "%)";
    } else {
        $search_row['best_error'] = number_format($search_row['best_error'], 3);
        $search_row['best_predictions'] = $bp . " (" . number_format(100.0 * $bp / 60000.00, 2) . "%)";
    }

    $tp = $search_row['test_predictions'];
    $gp = $search_row['generalizability_predictions'];
    $search_row['fitness'] = number_format($search_row['fitness'], 3);
    $search_row['generalizability_error'] = number_format($search_row['generalizability_error'], 3);
    $search_row['generalizability_predictions'] = $gp . " (" . number_format(100.0 * $gp / 5000.00, 2) . "%)";
    $search_row['test_error'] = number_format($search_row['test_error'], 3);
    $search_row['test_predictions'] = $tp . " (" . number_format(100.0 * $tp / 5000.00, 2) . "%)";

    $search_row['alpha'] = number_format($search_row['alpha'], 5);
    $search_row['initial_mu'] = number_format($search_row['initial_mu'], 5);
    $search_row['mu_delta'] = number_format($search_row['mu_delta'], 5);
    $search_row['initial_learning_rate'] = number_format($search_row['initial_learning_rate'], 10);
    $search_row['learning_rate_delta'] = number_format($search_row['learning_rate_delta'], 5);
    $search_row['initial_weight_decay'] = number_format($search_row['initial_weight_decay'], 10);
    $search_row['weight_decay_delta'] = number_format($search_row['weight_decay_delta'], 5);
    $search_row['input_dropout_probability'] = number_format($search_row['input_dropout_probability'], 5);
    $search_row['hidden_dropout_probability'] = number_format($search_row['hidden_dropout_probability'], 5);

    $stderr_result = query_multi_db($db_name, "SELECT ISNULL(stderr_out) FROM cnn_genome WHERE id = " . $search_row['id']);
    $stderr_row = $stderr_result->fetch_assoc();

    $search_row['db_name'] = $db_name;

    $search_rows['row'][] = $search_row;

    $search_result = query_multi_db($db_name, "SELECT id, best_error, best_predictions, generalizability_error, generalizability_predictions, test_error, test_predictions, best_error_epoch, max_epochs, generation_id, hyperparameters FROM cnn_genome WHERE exact_id = " . $search_id . " ORDER BY generalizability_error");

    while ($search_row = $search_result->fetch_assoc()) {
        $bp = $search_row['best_predictions'];

        if (strpos($search_name, 'cifar') !== false) {
            $search_row['best_error'] = number_format($search_row['best_error'], 3);
            $search_row['best_predictions'] = $bp . " (" . number_format(100.0 * $bp / 50000.00, 2) . "%)";
        } else {
            $search_row['best_error'] = number_format($search_row['best_error'], 3);
            $search_row['best_predictions'] = $bp . " (" . number_format(100.0 * $bp / 60000.00, 2) . "%)";
        }

        $tp = $search_row['test_predictions'];
        $gp = $search_row['generalizability_predictions'];
        $search_row['fitness'] = number_format($search_row['fitness'], 3);
        $search_row['generalizability_error'] = number_format($search_row['generalizability_error'], 3);
        $search_row['generalizability_predictions'] = $gp . " (" . number_format(100.0 * $gp / 5000.00, 2) . "%)";
        $search_row['test_error'] = number_format($search_row['test_error'], 3);
        $search_row['test_predictions'] = $tp . " (" . number_format(100.0 * $tp / 5000.00, 2) . "%)";

        $search_row['alpha'] = number_format($search_row['alpha'], 5);
        $search_row['initial_mu'] = number_format($search_row['initial_mu'], 5);
        $search_row['mu_delta'] = number_format($search_row['mu_delta'], 5);
        $search_row['initial_learning_rate'] = number_format($search_row['initial_learning_rate'], 10);
        $search_row['learning_rate_delta'] = number_format($search_row['learning_rate_delta'], 5);
        $search_row['initial_weight_decay'] = number_format($search_row['initial_weight_decay'], 10);
        $search_row['weight_decay_delta'] = number_format($search_row['weight_decay_delta'], 5);
        $search_row['input_dropout_probability'] = number_format($search_row['input_dropout_probability'], 5);
        $search_row['hidden_dropout_probability'] = number_format($search_row['hidden_dropout_probability'], 5);

        $stderr_result = query_multi_db($db_name, "SELECT ISNULL(stderr_out) FROM cnn_genome WHERE id = " . $search_row['id']);
        $stderr_row = $stderr_result->fetch_assoc();

        $search_row['db_name'] = $db_name;

        $search_rows['row'][] = $search_row;

    }

}

echo "
        <div class='row'>
            <div class='col-sm-12'>";

$projects_template = file_get_contents($cwd[__FILE__] . "/templates/exact_search.html");

$m = new Mustache_Engine;
echo $m->render($projects_template, $search_rows);

echo "
            </div> <!-- col-sm-12 -->
        </div> <!-- row -->

    </div> <!-- /container -->";


print_footer('Travis Desell and the Wildlife@Home Team', "Travis Desell</body></html>");

?>
