#include <cstdio>
#include <cstdlib>

#include <iostream>
using std::cout;
using std::endl;

#include <fstream>
using std::ifstream;
using std::ofstream;

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
    if (argc != 5) {
        cerr << "error: incorrect arguments." << endl;
        cerr << "usage: " << endl;
        cerr << "    " << argv[0] << " <mnist image file> <mnist label file> <output file> <expected number of images>" << endl;
        exit(1);
    }

    string image_filename(argv[1]);
    string label_filename(argv[2]);
    string output_filename(argv[3]);
    int expected_images = stoi(argv[4]);

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

    ofstream outfile;
    outfile.open(output_filename.c_str(), ios::out | ios::binary);

    vector<int> initial_vals;
    initial_vals.push_back(number_labels);
    initial_vals.push_back(number_channels);
    initial_vals.push_back(number_cols);
    initial_vals.push_back(number_rows);

    uint32_t sum = 0;
    for (int i = 0; i < images.size(); i++) {
        cout << "read " << images[i].size() << " images of class " << i << endl;
        initial_vals.push_back(images[i].size());
        sum += images[i].size();
    }
    cout << "read " << sum << " images in total" << endl;

    outfile.write( (char*)&initial_vals[0], initial_vals.size() * sizeof(int) );

    for (int i = 0; i < images.size(); i++) {
        for (int j = 0; j < images[i].size(); j++) {
            unsigned char pixel;

            for (int z = 0; z < number_channels; z++) {
                for (int y = 0; y < number_cols; y++) {
                    for (int x = 0; x < number_rows; x++) {
                        pixel = images[i][j][z][y][x];

                        outfile.write( (char*)&pixel, sizeof(char));

                        //                cout << " " << (uint32_t)pixel;
                    }
                    //            cout << endl;
                }
                //        cout << endl;
            }
        }

        cout << "wrote " << images[i].size() << " images." << endl;
    }

    outfile.close();

    return 0;
}

