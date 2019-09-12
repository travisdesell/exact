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

#include "image_tools/large_image_set.hxx"

int main(int argc, char **argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    string genome_filename;
    get_argument(arguments, "--genome_file", true, genome_filename);

    bool is_checkpoint = false;
    CNN_Genome *genome = new CNN_Genome(genome_filename, is_checkpoint);

    string label_name;
    get_argument(arguments, "--label_name", true, label_name);

    string label_type;
    get_argument(arguments, "--label_type", true, label_type);

    string mosaic_filename;
    get_argument(arguments, "--mosaic_filename", true, mosaic_filename);

    cout << "applying CNN '" << genome_filename << "' to mosaic '" << mosaic_filename << " using label '" << label_name << "' with type '" << label_type << "'" << endl;

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);

    if (label_type.compare("POINT") == 0) {
        int32_t padding = 2;
        int32_t subimage_y = 32;
        int32_t subimage_x = 32;

        LargeImages mosaic_image(mosaic_filename, padding, subimage_y, subimage_x);

        cout << endl << "drawing image predictions." << endl;

        ostringstream output_filename;
        output_filename << output_directory << "/test_output";

        LargeImage *image = mosaic_image.copy_image(0);
        image->draw_png(output_filename.str() + "_test.png");

        int stride = 1;
        vector< vector<float> > expanded_prediction_matrix;
        genome->get_expanded_prediction_matrix(mosaic_image, 0, stride, 0, expanded_prediction_matrix);
        cout << "got expandded prediction matrix" << endl;

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

        delete image;
        image = mosaic_image.copy_image(0);

        image->set_alpha(expanded_prediction_matrix);

        image->draw_png(output_filename.str() + "_original.png");
        image->draw_png_alpha(output_filename.str() + "_predictions.png");
        image->draw_png_4channel(output_filename.str() + "_merged.png");

        delete image;
    } 

}
