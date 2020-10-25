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

    if ($db_name == "exact_bn_pool") {
        $search_result = query_multi_db($db_name, "SELECT id, search_name, inserted_genomes, genomes_generated, max_genomes, max_epochs, best_predictions_genome_id FROM exact_search");
    } else {
        $search_result = query_multi_db($db_name, "SELECT id, search_name, inserted_genomes, genomes_generated, max_genomes, max_epochs FROM exact_search");
    }

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

        if ($db_name == "exact_bn_sfmp") {
            $fitness_result = query_multi_db($db_name, "SELECT best_validation_error, best_validation_predictions, best_epoch, training_error, training_predictions, test_error, test_predictions FROM cnn_genome WHERE exact_id = " . $search_row['id'] . " ORDER BY (best_validation_error) LIMIT 1");
        } else {
            $fitness_result = query_multi_db($db_name, "SELECT best_error, best_predictions, best_error_epoch, generalizability_error, generalizability_predictions, test_error, test_predictions FROM cnn_genome WHERE exact_id = " . $search_row['id'] . " ORDER BY (generalizability_error) LIMIT 1");
        }
        $fitness_row = $fitness_result->fetch_assoc();

        $search_row['best_error'] = number_format($fitness_row['best_error']);

        if (strpos($search_row['search_name'], "cifar") !== false) {
            $search_row['best_predictions'] = number_format(100 * $fitness_row['best_predictions'] / 50000, 2) . "%";
        } else {
            $search_row['best_predictions'] = number_format(100 * $fitness_row['best_predictions'] / 60000, 2) . "%";
        }

        if ($db_name == "exact_bn_sfmp") {
            $search_row['best_epoch'] = $fitness_row['best_epoch'];

            $search_row['training_error'] = number_format($fitness_row['training_error']);
            $search_row['training_predictions'] = number_format(100 * $fitness_row['training_predictions'] / 50000, 2) . "%";

            $search_row['best_validation_error'] = number_format($fitness_row['best_validation_error']);
            $search_row['best_validation_predictions'] = number_format($fitness_row['best_validation_predictions']);

        } else {
            $search_row['best_error_epoch'] = $fitness_row['best_error_epoch'];

            $search_row['generalizability_error'] = number_format($fitness_row['generalizability_error']);
            $search_row['generalizability_predictions'] = number_format(100 * $fitness_row['generalizability_predictions'] / 5000, 2) . "%";
        }

        $search_row['test_error'] = number_format($fitness_row['test_error']);
        $search_row['test_predictions'] = number_format(100 * $fitness_row['test_predictions'] / 5000, 2) . "%";

        if ($db_name == "exact_bn_sfmp") {
        } else if ($db_name == "exact_bn_pool") {
            if ($search_row['best_predictions_genome_id'] != 0) {
                $fitness_result = query_multi_db($db_name, "SELECT test_predictions, generalizability_predictions FROM cnn_genome WHERE exact_id = " . $search_row['id'] . " AND id = " . $search_row['best_predictions_genome_id']);
                $fitness_row = $fitness_result->fetch_assoc();

                $search_row['best_total_predictions'] = number_format(100 * ($fitness_row['test_predictions'] + $fitness_row['generalizability_predictions']) / 10000.0, 2) . "%";
            }
        } else {
            $fitness_result = query_multi_db($db_name, "SELECT test_predictions, generalizability_predictions FROM cnn_genome WHERE exact_id = " . $search_row['id'] . " ORDER BY (generalizability_predictions + test_predictions) DESC LIMIT 1");
            $fitness_row = $fitness_result->fetch_assoc();

            $search_row['best_total_predictions'] = number_format(100 * ($fitness_row['test_predictions'] + $fitness_row['generalizability_predictions']) / 10000.0, 2) . "%";
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

    if ($db_name == "exact_bn_sfmp") {
        $projects_template = file_get_contents($cwd[__FILE__] . "/templates/search_overview_new.html");

        $m = new Mustache_Engine;
        echo $m->render($projects_template, $search_rows);
    } else {
        $projects_template = file_get_contents($cwd[__FILE__] . "/templates/search_overview.html");

        $m = new Mustache_Engine;
        echo $m->render($projects_template, $search_rows);
    }


    echo "
            </div> <!-- col-sm-12 -->
        </div> <!-- row -->

    </div> <!-- /container -->";
}

//print_db_overview("exact");
print_db_overview("exact_batchnorm", "EXACT BATCH NORMALIZATION");
print_db_overview("exact_bn_pool", "EXACT BATCH NORMALIZATION WITH POOLING");

print_footer('Travis Desell and the Wildlife@Home Team', "Travis Desell</body></html>");

?>
