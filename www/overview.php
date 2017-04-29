<?php

$cwd[__FILE__] = __FILE__;
if (is_link($cwd[__FILE__])) $cwd[__FILE__] = readlink($cwd[__FILE__]);
$cwd[__FILE__] = dirname($cwd[__FILE__]);

require_once($cwd[__FILE__] . "/../../citizen_science_grid/header.php");
require_once($cwd[__FILE__] . "/../../citizen_science_grid/navbar.php");
require_once($cwd[__FILE__] . "/../../citizen_science_grid/footer.php");
require_once($cwd[__FILE__] . "/my_query.php");

print_header("Wildlife@Home: EXACT Search Progress", "<script type='text/javascript' src=''></script>", "wildlife");
print_navbar("Projects: Wildlife@Home", "Wildlife@Home", "..");

function print_db_overview($db_name, $title) {
    global $cwd;

    echo "
    <div class='container'>";

/*
echo "
        <div class='row' style='margin-bottom:10px;'>
            <div class='col-sm-12'>
                <button type='button' id='display-inactive-runs-button' class='btn btn-primary pull-right'>
                    Display Inactive Runs
                </button>
            </div> <!--col-sm-12 -->
        </div> <!-- row -->";
 */

    echo "<h4>$title</h4>
        <div class='row'>
            <div class='col-sm-12'>";

    $search_result = query_multi_db($db_name, "SELECT id, search_name, inserted_genomes, genomes_generated, max_genomes, max_epochs, sort_by_fitness FROM exact_search");

    while ($search_row = $search_result->fetch_assoc()) {
    /*
    echo $search_row["id"] . " " . $search_row["search_name"] . " ";
    echo $search_row["mu"] . " " . $search_row["mu_decay"] . " ";
    echo $search_row["learning_rate"] . " " . $search_row["learning_rate_decay"] . " ";
    echo $search_row["weight_decay"] . " " . $search_row["weight_decay_decay"] . " ";
    echo $search_row["inserted_genomes"] . " " . $search_row["genomes_generated"] . " ";
    echo "<br>";
     */

        //fitness query

        if ($search_row['sort_by_fitness'] == 1) {
            $fitness_result = query_multi_db($db_name, "SELECT best_error, best_predictions, best_error_epoch FROM cnn_genome WHERE exact_id = " . $search_row['id'] . " ORDER BY best_error LIMIT 1");
            $fitness_row = $fitness_result->fetch_assoc();

            $search_row['best_error'] = number_format($fitness_row['best_error']);

            if (strpos($search_row['search_name'], "cifar") !== false) {
                $search_row['best_predictions'] = number_format(100 * $fitness_row['best_predictions'] / 50000, 2) . "%";
            } else {
                $search_row['best_predictions'] = number_format(100 * $fitness_row['best_predictions'] / 60000, 2) . "%";
            }

            $search_row['best_error_epoch'] = $fitness_row['best_error_epoch'];

            $fitness_result = query_multi_db($db_name, "SELECT best_error, best_error_epoch FROM cnn_genome WHERE exact_id = " . $search_row['id'] . " ORDER BY best_error DESC LIMIT 1");
            $fitness_row = $fitness_result->fetch_assoc();

            $search_row['worst_error'] = number_format($fitness_row['best_error']);
            $search_row['worst_error_epoch'] = $fitness_row['best_error_epoch'];

            $fitness_result = query_multi_db($db_name, "SELECT test_error, test_predictions FROM cnn_genome WHERE exact_id = " . $search_row['id'] . " ORDER BY best_error LIMIT 1");
            $fitness_row = $fitness_result->fetch_assoc();

            $search_row['test_error'] = number_format($fitness_row['test_error']);
            $search_row['test_predictions'] = number_format(100 * $fitness_row['test_predictions'] / 10000, 2) . "%";
        } else {
            $fitness_result = query_multi_db($db_name, "SELECT best_error, best_predictions, best_error_epoch FROM cnn_genome WHERE exact_id = " . $search_row['id'] . " ORDER BY best_predictions DESC LIMIT 1");
            $fitness_row = $fitness_result->fetch_assoc();

            $search_row['best_error'] = number_format($fitness_row['best_error']);

            if (strpos($search_row['search_name'], "cifar") !== false) {
                $search_row['best_predictions'] = number_format(100 * $fitness_row['best_predictions'] / 50000, 2) . "%";
            } else {
                $search_row['best_predictions'] = number_format(100 * $fitness_row['best_predictions'] / 60000, 2) . "%";
            }

            $search_row['best_error_epoch'] = $fitness_row['best_error_epoch'];

            $fitness_result = query_multi_db($db_name, "SELECT best_error, best_error_epoch FROM cnn_genome WHERE exact_id = " . $search_row['id'] . " ORDER BY best_predictions LIMIT 1");
            $fitness_row = $fitness_result->fetch_assoc();

            $search_row['worst_error'] = number_format($fitness_row['best_error']);
            $search_row['worst_error_epoch'] = $fitness_row['best_error_epoch'];

            $fitness_result = query_multi_db($db_name, "SELECT test_error, test_predictions FROM cnn_genome WHERE exact_id = " . $search_row['id'] . " ORDER BY best_predictions DESC LIMIT 1");
            $fitness_row = $fitness_result->fetch_assoc();

            $search_row['test_error'] = number_format($fitness_row['test_error']);
            $search_row['test_predictions'] = number_format(100 * $fitness_row['test_predictions'] / 10000, 2) . "%";
        }

        $search_row['is_hidden'] = false;
        $search_row['db_name'] = $db_name;
        $search_rows['row'][] = $search_row;

    /*
    if ($search_row['samples'] == 0) $search_row['is_hidden'] = true;

    $walks_result = query_multi_db($db_name, "SELECT AVG(current_steps) FROM gibbs_walk WHERE search_id = " . $search_row['id']);
    $walks_row = $walks_result->fetch_assoc();

    $search_row['avg_progress'] = $walks_row['AVG(current_steps)'];

    $search_rows['row'][] = $search_row;
     */
    }

    $projects_template = file_get_contents($cwd[__FILE__] . "/templates/search_overview.html");

    $m = new Mustache_Engine;
    echo $m->render($projects_template, $search_rows);


    echo "
            </div> <!-- col-sm-12 -->
        </div> <!-- row -->

    </div> <!-- /container -->";
}

//print_db_overview("exact");
print_db_overview("exact2", "EXACT 2");
print_db_overview("exact_mnist", "EXACT MNIST");
print_db_overview("exact_mnist_batch", "EXACT MNIST BATCH");
print_db_overview("exact_mnist_batch2", "EXACT MNIST BATCH 2");
print_db_overview("exact_cifar", "EXACT CIFAR");


print_footer('Travis Desell and the Wildlife@Home Team', "Travis Desell</body></html>");

?>
