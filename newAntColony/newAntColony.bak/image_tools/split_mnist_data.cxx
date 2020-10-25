#include <algorithm>
using std::shuffle;

#include <cstdio>
#include <cstdlib>

#include <iostream>
using std::cout;
using std::endl;

#include <fstream>
using std::ifstream;
using std::ofstream;

#include <random>
using std::minstd_rand0;

#include <vector>
using std::vector;

using namespace std;

void flip_bytes(char *bytes, size_t size) {
    for (uint32_t i = 0; i < size / 2; i++) {
        char tmp = bytes[i];

        bytes[i] = bytes[(size - i) - 1];
        bytes[(size - i) - 1] = tmp;
    }
}

uint32_t fread_uint32_t(ifstream &file, string filename, const char *name, uint32_t expected_value) {
    uint32_t value;
    file.read( (char*)&value, sizeof(uint32_t) );
    flip_bytes( (char*)&value, sizeof(uint32_t) );

    cout << name << " is: " << value << endl;
    if (value != expected_value) {
        cerr << "Error reading from image file '" << filename << "'. Invalid format, " << name << " != " << expected_value << endl;
        exit(1);
    }

    return value;
}

int main(int argc, char** argv) {
    if (argc != 7) {
        cerr << "error: incorrect arguments." << endl;
        cerr << "usage: " << endl;
        cerr << "    " << argv[0] << " <mnist image file> <mnist label file> <output training file> <output validation file> <expected number of images> <validation images per label" << endl;
        exit(1);
    }

    string image_filename(argv[1]);
    string label_filename(argv[2]);
    string output_filename_test(argv[3]);
    string output_filename_validation(argv[4]);
    int expected_images = stoi(argv[5]);
    int images_per_label = stoi(argv[6]);

    ifstream image_file(image_filename.c_str(), ios::in | ios::binary);
    ifstream label_file(label_filename.c_str(), ios::in | ios::binary);

    if (!image_file.is_open()) {
        cerr << "Could not open '" << image_filename << "' for reading." << endl;
    }

    if (!label_file.is_open()) {
        cerr << "Could not open '" << label_filename << "' for reading." << endl;
    }

    fread_uint32_t(image_file, image_filename, "image file magic number", 2051);

    uint32_t number_images = fread_uint32_t(image_file, image_filename, "number images", expected_images);
    uint32_t number_rows = fread_uint32_t(image_file, image_filename, "number rows", 28);
    uint32_t number_cols = fread_uint32_t(image_file, image_filename, "number cols", 28);

    fread_uint32_t(label_file, label_filename, "label file magic number", 2049);

    uint32_t total_number_labels = fread_uint32_t(label_file, label_filename, "number labels", expected_images);


    if (number_images != total_number_labels) {
        cerr << "ERROR! Number images (" << number_images << ") != number labels (" << total_number_labels << ")" << endl;
        cerr << "make sure image and label files match!" << endl;
        exit(1);
    }

    uint32_t number_labels = 10;
    uint32_t number_channels = 1;

    vector< vector< vector< vector< vector<char> > > > > images(number_labels);
    vector< vector< vector< vector< vector<char> > > > > images_split(number_labels);

    unsigned char pixel;
    unsigned char label;
    for (uint32_t i = 0; i < number_images; i++) {
        vector< vector< vector<char> > > image(number_channels, vector< vector<char> >(number_cols, vector<char>(number_rows, 0)));

        for (uint32_t z = 0; z < number_channels; z++) {
            for (uint32_t y = 0; y < number_cols; y++) {
                for (uint32_t x = 0; x < number_rows; x++) {
                    image_file.read( (char*)&pixel, sizeof(char) );

                    //pixel = 255 - pixel;
                    image[z][y][x] = pixel;
                }
            }
        }

        label_file.read( (char*)&label, sizeof(char) );
        cout << "label: " << (int)label << endl;

        images[ (int)label ].push_back(image);

        //char keystroke;
        //cin >> keystroke;
    }

    image_file.close();
    label_file.close();

    //int images_per_label = (number_images / 2.0) / number_labels;
    cout << "validation file '" << output_filename_validation << " will have " << images_per_label << " images per label." << endl;


    minstd_rand0 generator = minstd_rand0(time(NULL));
    for (uint32_t i = 0; i < number_labels; i++) {
        shuffle(images[i].begin(), images[i].end(), generator);

        for (uint32_t j = 0; j < images_per_label; j++) {
            images_split[i].push_back( images[i].back() );
            images[i].pop_back();
        }
    }

    cout << "test file '" << output_filename_test << " has the following images per label: " << endl;
    for (uint32_t i = 0; i < number_labels; i++) {
        cout << "\timages[" << i << "].size(): " << images[i].size() << endl;
    }

    cout << "validation file '" << output_filename_validation << " has the following images per label: " << endl;
    for (uint32_t i = 0; i < number_labels; i++) {
        cout << "\timages_split[" << i << "].size(): " << images_split[i].size() << endl;
    }


    cout << "writing test file" << endl;
    ofstream test_outfile;
    test_outfile.open(output_filename_test.c_str(), ios::out | ios::binary);

    vector<int> initial_vals;
    initial_vals.push_back(number_labels);
    initial_vals.push_back(number_channels);
    initial_vals.push_back(number_cols);
    initial_vals.push_back(number_rows);

    uint32_t sum = 0;
    for (int i = 0; i < images.size(); i++) {
        cout << "\tread " << images[i].size() << " images of class " << i << endl;
        initial_vals.push_back(images[i].size());
        sum += images[i].size();
    }
    cout << "read " << sum << " images in total" << endl;

    test_outfile.write( (char*)&initial_vals[0], initial_vals.size() * sizeof(int) );

    for (int i = 0; i < images.size(); i++) {
        for (int j = 0; j < images[i].size(); j++) {
            unsigned char pixel;

            for (int z = 0; z < number_channels; z++) {
                for (int y = 0; y < number_cols; y++) {
                    for (int x = 0; x < number_rows; x++) {
                        pixel = images[i][j][z][y][x];

                        test_outfile.write( (char*)&pixel, sizeof(char));

                        //                cout << " " << (uint32_t)pixel;
                    }
                    //            cout << endl;
                }
                //        cout << endl;
            }
        }

        cout << "\twrote " << images[i].size() << " images." << endl;
    }

    test_outfile.close();

    cout << "writing validation file" << endl;
    ofstream validation_outfile;
    validation_outfile.open(output_filename_validation.c_str(), ios::out | ios::binary);

    initial_vals.clear();
    initial_vals.push_back(number_labels);
    initial_vals.push_back(number_channels);
    initial_vals.push_back(number_cols);
    initial_vals.push_back(number_rows);

    sum = 0;
    for (int i = 0; i < images_split.size(); i++) {
        cout << "\tread " << images_split[i].size() << " images_split of class " << i << endl;
        initial_vals.push_back(images_split[i].size());
        sum += images_split[i].size();
    }
    cout << "read " << sum << " images_split in total" << endl;

    validation_outfile.write( (char*)&initial_vals[0], initial_vals.size() * sizeof(int) );

    for (int i = 0; i < images_split.size(); i++) {
        for (int j = 0; j < images_split[i].size(); j++) {
            unsigned char pixel;

            for (int z = 0; z < number_channels; z++) {
                for (int y = 0; y < number_cols; y++) {
                    for (int x = 0; x < number_rows; x++) {
                        pixel = images_split[i][j][z][y][x];

                        validation_outfile.write( (char*)&pixel, sizeof(char));

                        //                cout << " " << (uint32_t)pixel;
                    }
                    //            cout << endl;
                }
                //        cout << endl;
            }
        }

        cout << "\twrote " << images_split[i].size() << " images_split." << endl;
    }

    validation_outfile.close();
    return 0;
}

