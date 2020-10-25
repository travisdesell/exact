#include <iomanip>
using std::setw;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"

#ifdef _MYSQL_
#include "common/db_conn.hxx"
#endif

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "cnn/exact.hxx"
#include "cnn/cnn_genome.hxx"
#include "cnn/cnn_edge.hxx"
#include "cnn/cnn_node.hxx"

#include "image_tools/mosaic_image_set.hxx"

int main(int argc, char **argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    string genome_filename;
    get_argument(arguments, "--genome_file", true, genome_filename);

    bool is_checkpoint = false;
    CNN_Genome *genome = new CNN_Genome(genome_filename, is_checkpoint);

    string db_file;
    get_argument(arguments, "--db_file", true, db_file);
    set_db_info_filename(db_file);

    int mosaic_id;
    get_argument(arguments, "--mosaic_id", true, mosaic_id);

    int label_id;
    get_argument(arguments, "--label_id", true, label_id);

    //get the mosaic info
    ostringstream mosaic_query;
    mosaic_query << "SELECT filename, height, width FROM mosaics WHERE id = " << mosaic_id;
    mysql_exact_query(mosaic_query.str());

    MYSQL_RES *mosaic_result = mysql_store_result(exact_db_conn);

    if (mosaic_result == NULL) {
        cerr << "ERROR: no label in database with id " << label_id << endl;
        exit(1);
    }

    MYSQL_ROW mosaic_row = mysql_fetch_row(mosaic_result);

    string mosaic_filename = mosaic_row[0];
    int32_t mosaic_height = atoi(mosaic_row[1]);
    int32_t mosaic_width = atoi(mosaic_row[2]);


    //get the label info
    ostringstream label_query;
    label_query << "SELECT label_name, label_type FROM labels WHERE label_id = " << label_id;
    mysql_exact_query(label_query.str());

    MYSQL_RES *label_result = mysql_store_result(exact_db_conn);

    if (label_result == NULL) {
        cerr << "ERROR: no label in database with id " << label_id << endl;
        exit(1);
    }

    MYSQL_ROW label_row = mysql_fetch_row(label_result);

    string label_name = label_row[0];
    string label_type = label_row[1];

    cout << "applying CNN '" << genome_filename << "' to mosaic '" << mosaic_filename << " using label '" << label_name << "' with type '" << label_type << "'" << endl;

    string input_directory;
    get_argument(arguments, "--input_directory", true, input_directory);

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);

    int owner_id;
    get_argument(arguments, "--owner_id", true, owner_id);

    string job_name;
    get_argument(arguments, "--job_name", true, job_name);

    input_directory += to_string(owner_id) + "/";
    output_directory += to_string(owner_id) + "/";


    vector<string> filenames;
    filenames.push_back(input_directory + mosaic_filename);

    if (label_type.compare("POINT") == 0) {
        vector< vector< Point > > points;
        vector< Point > mosaic_points;

        ostringstream point_query;
        point_query << "SELECT point_id, cx, cy FROM `points` WHERE mosaic_id = " << mosaic_id << " AND label_id = " << label_id << endl;
        mysql_exact_query(point_query.str());

        MYSQL_RES *point_result = mysql_store_result(exact_db_conn);

        if (point_result == NULL) {
            cerr << "ERROR: no points in database with mosaic_id " << mosaic_id << " and label_id " << label_id << endl;
            exit(1);
        }

        MYSQL_ROW point_row;
        vector<int32_t> point_ids;
        while ((point_row = mysql_fetch_row(point_result)) != NULL) {
            int32_t point_id = atoi(point_row[0]);
            point_ids.push_back(point_id);

            int32_t multiplier = fmax(mosaic_width, mosaic_height);
            int32_t cx = atof(point_row[1]) * multiplier;
            int32_t cy = atof(point_row[2]) * multiplier;

            cout << "adding point " << point_id << ": " << cx << ", " << cy << endl;

            mosaic_points.push_back(Point(cy, cx));
        }

        points.push_back(mosaic_points);


        vector< vector<int> > point_classes;
        vector<int> mosaic_point_classes;
        for (uint32_t i = 0; i < mosaic_points.size(); i++) {
            mosaic_point_classes.push_back(1);
        }
        point_classes.push_back(mosaic_point_classes);

        int point_radius = 64;
        int32_t padding = 2;
        int32_t subimage_y = 32;
        int32_t subimage_x = 32;

        MosaicImages point_mosaic_images(filenames, points, point_radius, point_classes, padding, subimage_y, subimage_x);

        cout << endl << "drawing image predictions." << endl;
        //genome->draw_predictions(line_mosaic_images, output_directory);

        ostringstream job_query;
        job_query << "INSERT INTO jobs SET owner_id = " << owner_id << ", mosaic_id = " << mosaic_id << ", label_id = " << label_id << ", name = '" << job_name << "'";
        mysql_exact_query(job_query.str());
        
        int job_id = mysql_exact_last_insert_id();


        int stride = 1;
        for (uint32_t i = 0; i < point_mosaic_images.get_number_large_images(); i++) {
            cout << endl << endl;
            cout << "PROCESSING IMAGE " << i << endl;

            vector< vector<float> > expanded_prediction_matrix;
            genome->get_expanded_prediction_matrix(point_mosaic_images, i, stride, 0, expanded_prediction_matrix);

            float max_prediction = 0.0;
            int32_t max_y = 0, max_x = 0;
            for (uint32_t y = 0; y < expanded_prediction_matrix.size(); y++) {
                for (uint32_t x = 0; x < expanded_prediction_matrix[y].size(); x++) {
                    if (expanded_prediction_matrix[y][x] > max_prediction) {
                        max_prediction = expanded_prediction_matrix[y][x];
                        max_y = y;
                        max_x = x;
                    }
                }
            }

            cout << "prediction: " << max_prediction << endl;

            ostringstream prediction_query;
            prediction_query << "INSERT INTO prediction SET job_id = " << job_id << ", owner_id = " << owner_id << ", mosaic_id = " << mosaic_id << ", label_id = " << label_id << ", mark_id = " << point_ids[i] << ", prediction = " << max_prediction;
            mysql_exact_query(prediction_query.str());


            LargeImage *image = point_mosaic_images.copy_image(i);

            for (int32_t y = max_y - 5; y <= max_y + 5; y++) {
                if (y < 0 || y >= 2 * point_radius) continue;
                image->set_pixel(0, y, max_x, (uint8_t)255);
                image->set_pixel(1, y, max_x, (uint8_t)0);
                image->set_pixel(2, y, max_x, (uint8_t)0);
            }

            for (int32_t x = max_x - 5; x <= max_x + 5; x++) {
                if (x < 0 || x >= 2 * point_radius) continue;
                image->set_pixel(0, max_y, x, (uint8_t)255);
                image->set_pixel(1, max_y, x, (uint8_t)0);
                image->set_pixel(2, max_y, x, (uint8_t)0);
            }

            image->set_alpha(expanded_prediction_matrix);

            ostringstream file_directory;
            file_directory << output_directory << job_id;
            mkdir(file_directory.str().c_str(), 0777);

            ostringstream output_filename;
            output_filename << output_directory << job_id << "/point_" << mosaic_id << "_" << label_id << "_" << point_ids[i];

            image->draw_png(output_filename.str() + "_original.png");
            image->draw_png_alpha(output_filename.str() + "_predictions.png");
            image->draw_png_4channel(output_filename.str() + "_merged.png");

            delete image;
        }

    } else if (label_type.compare("LINE") == 0) {
        vector< vector< Line > > lines;
        vector< Line > mosaic_lines;

        ostringstream line_query;
        line_query << "SELECT line_id, x1, y1, x2, y2 FROM `lines` WHERE mosaic_id = " << mosaic_id << " AND label_id = " << label_id << endl;
        mysql_exact_query(line_query.str());

        MYSQL_RES *line_result = mysql_store_result(exact_db_conn);

        if (line_result == NULL) {
            cerr << "ERROR: no lines in database with mosaic_id " << mosaic_id << " and label_id " << label_id << endl;
            exit(1);
        }

        MYSQL_ROW line_row;
        vector<int32_t> line_ids;
        while ((line_row = mysql_fetch_row(line_result)) != NULL) {
            int32_t line_id = atoi(line_row[0]);
            line_ids.push_back(line_id);

            int32_t multiplier = fmax(mosaic_width, mosaic_height);
            int32_t x1 = atof(line_row[1]) * multiplier;
            int32_t y1 = atof(line_row[2]) * multiplier;
            int32_t x2 = atof(line_row[3]) * multiplier;
            int32_t y2 = atof(line_row[4]) * multiplier;

            cout << "adding line " << line_id << ": " << x1 << ", " << y1 << ", " << x2 << ", " << y2 << endl;

            mosaic_lines.push_back(Line(y1, x1, y2, x2));
        }

        lines.push_back(mosaic_lines);

        int line_height = 64;

        vector< vector<int> > line_classes;
        vector<int> mosaic_line_classes;
        for (uint32_t i = 0; i < mosaic_lines.size(); i++) {
            mosaic_line_classes.push_back(1);
        }
        line_classes.push_back(mosaic_line_classes);

        int32_t padding = 2;
        int32_t subimage_y = 64;
        int32_t subimage_x = 64;

        MosaicImages line_mosaic_images(filenames, lines, line_height, line_classes, padding, subimage_y, subimage_x);

        cout << endl << "drawing image predictions." << endl;
        //genome->draw_predictions(line_mosaic_images, output_directory);

        ostringstream job_query;
        job_query << "INSERT INTO jobs SET owner_id = " << owner_id << ", mosaic_id = " << mosaic_id << ", label_id = " << label_id << ", name = '" << job_name << "'";
        mysql_exact_query(job_query.str());
        
        int job_id = mysql_exact_last_insert_id();

        int stride = 1;
        for (uint32_t i = 0; i < line_mosaic_images.get_number_large_images(); i++) {
            cout << endl << endl;
            cout << "PROCESSING IMAGE " << i << endl;

            vector< vector< vector<float> > > prediction_matrix;

            genome->get_prediction_matrix(line_mosaic_images, i, stride, prediction_matrix);

            float prediction = 0.0;
            int32_t count = 0;
            for (uint32_t y = 0; y < prediction_matrix.size(); y++) {
                for (uint32_t x = 0; x < prediction_matrix[y].size(); x++) {
                    /*
                    for (uint32_t c = 0; c < prediction_matrix[y][x].size(); c++) {
                        cout << " " << prediction_matrix[y][x][c];
                    }
                    cout << endl;
                    */

                    prediction += prediction_matrix[y][x][0];
                    count++;
                }
            }
            prediction /= count;
            cout << "prediction: " << prediction << endl;

            ostringstream prediction_query;
            prediction_query << "INSERT INTO prediction SET job_id = " << job_id << ", owner_id = " << owner_id << ", mosaic_id = " << mosaic_id << ", label_id = " << label_id << ", mark_id = " << line_ids[i] << ", prediction = " << prediction;
            mysql_exact_query(prediction_query.str());

            LargeImage *image = line_mosaic_images.copy_image(i);

            for (uint32_t j = 0; j < prediction_matrix[0].size(); j++) {
                int x = j + (image->get_height() / 2) + padding;

                int y = padding;
                image->set_pixel(0, y, x, 255 - (int8_t)(prediction_matrix[0][j][0] * 255.0));
                image->set_pixel(1, y, x, (int8_t)(prediction_matrix[0][j][0] * 255.0));
                image->set_pixel(2, y, x, 0);

                y = padding + 1;
                image->set_pixel(0, y, x, 255 - (int8_t)(prediction_matrix[0][j][0] * 255.0));
                image->set_pixel(1, y, x, (int8_t)(prediction_matrix[0][j][0] * 255.0));
                image->set_pixel(2, y, x, 0);

                y = image->get_height() - 1 + padding;
                image->set_pixel(0, y, x, 255 - (int8_t)(prediction_matrix[0][j][0] * 255.0));
                image->set_pixel(1, y, x, (int8_t)(prediction_matrix[0][j][0] * 255.0));
                image->set_pixel(2, y, x, 0);

                y = image->get_height() - 2 + padding;
                image->set_pixel(0, y, x, 255 - (int8_t)(prediction_matrix[0][j][0] * 255.0));
                image->set_pixel(1, y, x, (int8_t)(prediction_matrix[0][j][0] * 255.0));
                image->set_pixel(2, y, x, 0);
            }

            ostringstream file_directory;
            file_directory << output_directory << job_id;
            mkdir(file_directory.str().c_str(), 0777);

            ostringstream output_filename;
            output_filename << output_directory << job_id << "/line_" << mosaic_id << "_" << label_id << "_" << line_ids[i] << ".png";
            image->draw_png(output_filename.str().c_str());


            delete image;
        }
    } 

}
