<?php

$cwd[__FILE__] = __FILE__;
if (is_link($cwd[__FILE__])) $cwd[__FILE__] = readlink($cwd[__FILE__]);
$cwd[__FILE__] = dirname($cwd[__FILE__]);

require_once($cwd[__FILE__] . "/../../citizen_science_grid/my_query.php");
require_once($cwd[__FILE__] . "/../www/my_query.php");


$png_files = array();
$gv_files = array();

function process_genomes($db_name) {
    global $png_files, $gv_files;

    $genome_result = query_multi_db($db_name, "SELECT id, exact_id, test_error FROM cnn_genome");

    while ($genome_row = $genome_result->fetch_assoc()) {

        $exact_result = query_multi_db($db_name, "SELECT search_name, training_filename FROM exact_search WHERE id = " . $genome_row['exact_id']);
        $exact_row = $exact_result->fetch_assoc();
        $search_name = $exact_row['search_name'];
        $training_filename = $exact_row['training_filename'];
        $testing_filename = str_replace("training", "testing", $training_filename);

        $genome_id = $genome_row['id'];
        $genome_image = "/home/tdesell/exact/www/networks/" . $search_name . "_genome_" . $genome_id . ".png";
        $graphviz_file = "/home/tdesell/exact/www/networks/" . $search_name . "_genome_" . $genome_id . ".gv";

        $png_files[] = $search_name . "_genome_" . $genome_id . ".png";
        $gv_files[] = $search_name . "_genome_" . $genome_id . ".gv";

        if ($genome_row['test_error'] == 10000000) {
            $command = "/home/tdesell/exact/build/tests/evaluate_cnn --genome_id $genome_id --training_data $training_filename --testing_data $testing_filename --update_database --db_file /home/tdesell/exact/" . $db_name . "_db_info";
            echo "command: $command \n";
            echo "results: " . exec($command) . "\n";
        } else {
            echo "SKIPPING genome $genome_id test_error already calculated!\n";
        }

        if (!file_exists($genome_image)) {
            echo "'$graphviz_file' does not exist, generating\n";
            $command = "/home/tdesell/exact/build/tests/generate_gv $genome_id $graphviz_file /home/tdesell/exact/" . $db_name . "_db_info";
            echo "command: $command \n";
            echo "results: " . exec($command) . "\n";

            $command = "dot -Tpng $graphviz_file -o $genome_image";
            echo "command: $command \n";
            echo "results: " . exec($command) . "\n\n";

        } else {
            echo "SKIPPING '$genome_image', already created!\n";
        }
    }
}

process_genomes("exact_batchnorm");
process_genomes("exact_bn_pool");
process_genomes("exact_bn_sfmp");

echo "listing files!\n";

$dir = new DirectoryIterator("/home/tdesell/exact/www/networks/");
foreach ($dir as $fileinfo) {
    if (!$fileinfo->isDot()) {
        $filename = $fileinfo->getFilename();

        if (in_array($filename, $png_files) || in_array($filename, $gv_files)) {
            echo "$filename was in populations!\n";
        } else {
            echo "$filename WAS NOT IN populations!\n";
            $command = "rm /home/tdesell/exact/www/networks/" . $filename;
            echo "results: " . exec($command) . "\n";
        }
    }
}


?>

