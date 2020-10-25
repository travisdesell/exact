<?php

$cwd[__FILE__] = __FILE__;
if (is_link($cwd[__FILE__])) $cwd[__FILE__] = readlink($cwd[__FILE__]);
$cwd[__FILE__] = dirname($cwd[__FILE__]);

require_once($cwd[__FILE__] . "/../../citizen_science_grid/header.php");
require_once($cwd[__FILE__] . "/../../citizen_science_grid/navbar.php");
require_once($cwd[__FILE__] . "/../../citizen_science_grid/footer.php");
require_once($cwd[__FILE__] . "/my_query.php");

$genome_id = $boinc_db->real_escape_string($_GET['id']);
$db_name = $boinc_db->real_escape_string($_GET['db']);

print_header("Wildlife@Home: EXACT Search Progress", "<script type='text/javascript' src=''></script>", "wildlife");
print_navbar("Projects: Wildlife@Home", "Wildlife@Home", "..");

$genome_result = query_multi_db($db_name, "SELECT exact_id, generation_id FROM cnn_genome WHERE id = $genome_id");
$genome_row = $genome_result->fetch_assoc();
$search_id = $genome_row['exact_id'];
$generation_id = $genome_row['generation_id'];

$exact_result = query_multi_db($db_name, "SELECT search_name FROM exact_search WHERE id = $search_id");
$exact_row = $exact_result->fetch_assoc();
$search_name = $exact_row['search_name'];


echo "
    <div class='container'>";

echo "<h3>Genome $generation_id</h3>";

echo "
        <div class='row' style='margin-bottom:10px;'>
            <div class='col-sm-12'>";

if (file_exists($cwd[__FILE__] . "/networks/" . $search_name . "_genome_" . $genome_id . ".png")) {
    echo "<img src='./networks/" . $search_name . "_genome_" . $genome_id . ".png' width='100%'></img>";
} else {
    echo "Image of genome has not been generated yet, try back in a couple minutes.<br>";
    echo "expected filename: '" . $cwd[__FILE__] . "/networks/" . $search_name . "_genome_" . $genome_id . ".png'<br>";
}

echo "
            </div> <!--col-sm-12 -->
        </div> <!-- row -->";

echo "
        <div class='row'>
            <div class='col-sm-12'>";

$genome_result = query_multi_db($db_name, "SELECT * FROM cnn_genome WHERE id = " . $genome_id);
$genome_row = $genome_result->fetch_assoc();

$bp = $genome_row['best_predictions'];
$genome_row['best_error'] = number_format($genome_row['best_error'], 3);

if (strpos($search_name, "cifar") !== false) {
    $genome_row['best_predictions'] = $bp . " (" . number_format(100.0 * $bp / 50000.00, 2) . "%)";
} else {
    $genome_row['best_predictions'] = $bp . " (" . number_format(100.0 * $bp / 60000.00, 2) . "%)";
}

$tp = $genome_row['test_predictions'];
$genome_row['test_error'] = number_format($genome_row['test_error'], 3);
$genome_row['test_predictions'] = $tp . " (" . number_format(100.0 * $tp / 10000.00, 2) . "%)";

$gp = $genome_row['generalizability_predictions'];
$genome_row['generalizability_error'] = number_format($genome_row['generalizability_error'], 3);
$genome_row['generalizability_predictions'] = $gp . " (" . number_format(100.0 * $gp / 10000.00, 2) . "%)";

$genome_row['combined_error'] = $genome_row['generalizability_error'] + $genome_row['test_error'];
$genome_row['combined_predictions'] = $genome_row['generalizability_predictions'] + $genome_row['test_predictions'];
$cp = $genome_row['combined_predictions'];
$genome_row['combined_predictions'] = $cp . " (" . number_format(100.0 * $cp / 10000.00, 2) . "%)";


$node_result = query_multi_db($db_name, "SELECT count(*) FROM cnn_node WHERE genome_id = $genome_id AND disabled = 0");
$node_row = $node_result->fetch_assoc();
$genome_row['number_enabled_nodes'] = $node_row['count(*)'];

$node_result = query_multi_db($db_name, "SELECT count(*) FROM cnn_node WHERE genome_id = $genome_id AND disabled = 1");
$node_row = $node_result->fetch_assoc();
$genome_row['number_disabled_nodes'] = $node_row['count(*)'];



$edge_result = query_multi_db($db_name, "SELECT count(*) FROM cnn_edge WHERE genome_id = $genome_id AND disabled = 0 AND type = 0");
$edge_row = $edge_result->fetch_assoc();
$genome_row['number_enabled_conv_edges'] = $edge_row['count(*)'];

$edge_result = query_multi_db($db_name, "SELECT count(*) FROM cnn_edge WHERE genome_id = $genome_id AND disabled = 1 AND type = 0");
$edge_row = $edge_result->fetch_assoc();
$genome_row['number_disabled_conv_edges'] = $edge_row['count(*)'];

$edge_result = query_multi_db($db_name, "SELECT count(*) FROM cnn_edge WHERE genome_id = $genome_id AND disabled = 0 AND type = 1");
$edge_row = $edge_result->fetch_assoc();
$genome_row['number_enabled_pool_edges'] = $edge_row['count(*)'];

$edge_result = query_multi_db($db_name, "SELECT count(*) FROM cnn_edge WHERE genome_id = $genome_id AND disabled = 1 AND type = 1");
$edge_row = $edge_result->fetch_assoc();
$genome_row['number_disabled_pool_edges'] = $edge_row['count(*)'];



$weight_result = query_multi_db($db_name, "SELECT sum(filter_x * filter_y) FROM cnn_edge WHERE genome_id = $genome_id AND disabled = 0 AND type = 0");
$weight_row = $weight_result->fetch_assoc();
$weight_count = $weight_row['sum(filter_x * filter_y)'];
$genome_row['weight_count'] = $weight_count;


$stderr_result = query_multi_db($db_name, "SELECT stderr_out FROM cnn_genome WHERE id = $genome_id");
$stderr_row = $stderr_result->fetch_assoc();
$genome_row['stderr_out'] = $stderr_row['stderr_out'];

$projects_template = file_get_contents($cwd[__FILE__] . "/templates/genome.html");

$m = new Mustache_Engine;
echo $m->render($projects_template, $genome_row);

echo "
            </div> <!-- col-sm-12 -->
        </div> <!-- row -->

    </div> <!-- /container -->";


print_footer('Travis Desell and the Wildlife@Home Team', "Travis Desell</body></html>");

?>
