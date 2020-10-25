<?php

$cwd[__FILE__] = __FILE__;
if (is_link($cwd[__FILE__])) $cwd[__FILE__] = readlink($cwd[__FILE__]);
$cwd[__FILE__] = dirname($cwd[__FILE__]);

require_once($cwd[__FILE__] . "/../../citizen_science_grid/header.php");
require_once($cwd[__FILE__] . "/../../citizen_science_grid/navbar.php");
require_once($cwd[__FILE__] . "/../../citizen_science_grid/footer.php");
require_once($cwd[__FILE__] . "/../../citizen_science_grid/my_query.php");

print_header("Wildlife@Home: EXACT 'Monster' Workunits", "<script type='text/javascript' src=''></script>", "wildlife");
print_navbar("Projects: Wildlife@Home", "Wildlife@Home", "..");

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

echo "
        <div class='row'>
            <div class='col-sm-12'>";

$multiplier = 4;

$result = query_boinc_db("select r1.id as r1_id, r1.cpu_time as r1_cpu_time, r1.outcome as r1_outcome, r2.id as r2_id, r2.cpu_time as r2_cpu_time, r2.outcome as r2_outcome, r1.workunitid as r1_workunitid from result r1, result r2 where r1.cpu_time > (r2.cpu_time * $multiplier) and r1.workunitid = r2.workunitid and r1.cpu_time != 0 and r2.cpu_time != 0 and r1.id != r2.id and r1.outcome = 1 and r2.outcome = 1");


echo "
<table class='table table-striped table-bordered'>
    <thead>
        <th>Workunit ID</th>
        <th>Result 1 ID</th>
        <th>Result 2 ID</th>
        <th>Result 1 Outcome</th>
        <th>Result 2 Outcome</th>
        <th>Result 1 CPU Time</th>
        <th>Result 2 CPU Time</th>
    </thead>

    <tbody>";

while ($row = $result->fetch_assoc()) {
    //echo json_encode($row) . "<br>";

    $r1_id = $row['r1_id'];
    $r1_cpu_time = $row['r1_cpu_time'];
    $r1_outcome = $row['r1_outcome'];
    $r2_id = $row['r2_id'];
    $r2_cpu_time = $row['r2_cpu_time'];
    $r2_outcome = $row['r2_outcome'];
    $workunit_id = $row['r1_workunitid'];


    echo "<tr>";
    echo "<td><a href='http://csgrid.org/csg/workunit.php?wuid=$workunit_id'>$workunit_id</a></td>";
    echo "<td><a href='http://csgrid.org/csg/result.php?resultid=$r1_id'>$r1_id</a></td>";
    echo "<td><a href='http://csgrid.org/csg/result.php?resultid=$r2_id'>$r2_id</a></td>";
    echo "<td>$r1_outcome</td>";
    echo "<td>$r2_outcome</td>";
    echo "<td>$r1_cpu_time</td>";
    echo "<td>$r2_cpu_time</td>";
    echo "</tr>";
}

echo "</tbody></table>";


echo "
            </div> <!-- col-sm-12 -->
        </div> <!-- row -->

    </div> <!-- /container -->";


print_footer('Travis Desell and the Wildlife@Home Team', "Travis Desell</body></html>");

?>
