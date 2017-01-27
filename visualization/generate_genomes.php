<?php

$cwd[__FILE__] = __FILE__;
if (is_link($cwd[__FILE__])) $cwd[__FILE__] = readlink($cwd[__FILE__]);
$cwd[__FILE__] = dirname($cwd[__FILE__]);

require_once($cwd[__FILE__] . "/../../citizen_science_grid/my_query.php");

$genome_result = query_boinc_db("SELECT id, exact_id FROM cnn_genome");

while ($genome_row = $genome_result->fetch_assoc()) {

    $exact_result = query_boinc_db("SELECT search_name FROM exact_search WHERE id = " . $genome_row['exact_id']);
    $exact_row = $exact_result->fetch_assoc();
    $search_name = $exact_row['search_name'];

    $genome_id = $genome_row['id'];
    $genome_image = "/home/tdesell/exact/www/networks/" . $search_name . "_genome_" . $genome_id . ".png";
    $graphviz_file = "/home/tdesell/exact/www/networks/" . $search_name . "_genome_" . $genome_id . ".gv";

    if (!file_exists($genome_image)) {
        echo "'$graphviz_file' does not exist, generating\n";
        $command = "/home/tdesell/exact/build/tests/generate_gv $genome_id $graphviz_file";
        echo "command: $command \n";
        echo "results: " . exec($command) . "\n";

        $command = "dot -Tpng $graphviz_file -o $genome_image";
        echo "command: $command \n";
        echo "results: " . exec($command) . "\n\n";

    } else {
        echo "SKIPPING '$genome_image', already created!\n";
    }
}

?>

