<?php

$cwd[__FILE__] = __FILE__;
if (is_link($cwd[__FILE__])) $cwd[__FILE__] = readlink($cwd[__FILE__]);
$cwd[__FILE__] = dirname($cwd[__FILE__]);

require_once($cwd[__FILE__] . "/../../citizen_science_grid/my_query.php");

$genome_result = query_boinc_db("SELECT id, exact_id, ISNULL(test_error) FROM cnn_genome");

$png_files = array();
$gv_files = array();

while ($genome_row = $genome_result->fetch_assoc()) {

    $exact_result = query_boinc_db("SELECT search_name FROM exact_search WHERE id = " . $genome_row['exact_id']);
    $exact_row = $exact_result->fetch_assoc();
    $search_name = $exact_row['search_name'];

    $genome_id = $genome_row['id'];
    $genome_image = "/home/tdesell/exact/www/networks/" . $search_name . "_genome_" . $genome_id . ".png";
    $graphviz_file = "/home/tdesell/exact/www/networks/" . $search_name . "_genome_" . $genome_id . ".gv";

    $png_files[] = $search_name . "_genome_" . $genome_id . ".png";
    $gv_files[] = $search_name . "_genome_" . $genome_id . ".gv";

    if ($genome_row['ISNULL(test_error)'] == 1) {
        $command = "/home/tdesell/exact/build/tests/evaluate_cnn --genome_id $genome_id --training_data /home/tdesell/exact/datasets/mnist_training_data.bin --testing_data /home/tdesell/exact/datasets/mnist_testing_data.bin --update_database --db_file /home/tdesell/exact/exact_db_info";
        echo "command: $command \n";
        echo "results: " . exec($command) . "\n";
    } else {
        echo "SKIPPING genome $genome_id test_error already calculated!";
    }

    if (!file_exists($genome_image)) {
        echo "'$graphviz_file' does not exist, generating\n";
        $command = "/home/tdesell/exact/build/tests/generate_gv $genome_id $graphviz_file /projects/csg/exact_db_info";
        echo "command: $command \n";
        echo "results: " . exec($command) . "\n";

        $command = "dot -Tpng $graphviz_file -o $genome_image";
        echo "command: $command \n";
        echo "results: " . exec($command) . "\n\n";

    } else {
        echo "SKIPPING '$genome_image', already created!\n";
    }
}

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

