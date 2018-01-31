<?php

//$db_info_file = file_get_contents("../exact_batchnorm_db_info");
$db_info_file = file_get_contents("../exact_bn_sfmp_db_info");
$db_info_file_lines = explode("\n", $db_info_file);

echo "connecting to database with information:\n";

foreach ($db_info_file_lines as $line) {
    echo $line . "\n";
}

$exact_host = $db_info_file_lines[0];
$exact_db_name = $db_info_file_lines[1];
$exact_user = $db_info_file_lines[2];
$exact_passwd = $db_info_file_lines[3];

$exact_db = NULL;

function connect_exact_db() {
    global $exact_db, $exact_user, $exact_passwd, $exact_host, $exact_db_name;

    $exact_db = new mysqli($exact_host, $exact_user, $exact_passwd, $exact_db_name);
    if ($exact_db->connect_errno) {
        echo "Failed to connect to MySQL: (" . $exact_db->connect_errno . ") " . $exact_db->connect_error . "\n";
    }
}

connect_exact_db();

function mysqli_error_msg($db, $query, $die_on_error = true) {
    error_log("MYSQL Error (" . $db->errno . "): " . $db->error . ", query: $query" . "\n");

    if ($die_on_error) die("MYSQL Error (" . $db->errno . "): " . $db->error . ", query: $query" . "\n");
}


function query_exact_db($query, $die_on_error = true) {
    global $exact_db;

    if (!$exact_db->ping()) connect_exact_db();

    $result = $exact_db->query($query);

    if (!$result) mysqli_error_msg($exact_db, $query, $die_on_error);

    return $result;
}

function insert_id_exact_db() {
    global $exact_db;

    return mysqli_insert_id($exact_db);
}

?>
