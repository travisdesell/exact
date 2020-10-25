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

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "error: incorrect arguments." << endl;
        cerr << "usage: " << endl;
        cerr << "    " << argv[0] << " <cifar 10 image files> <output file> <expected number of images per file>" << endl;
        exit(1);
    }

    vector<string> image_filenames;
    for (uint32_t i = 1; i < argc - 2; i++) {
        image_filenames.push_back(argv[i]);
        cout << "input files: " << image_filenames.back() << endl;
    }
    
    string output_filename(argv[argc - 2]);
    int expected_images = stoi(argv[argc - 1]);

    int number_labels = 10;
    int number_channels = 3;
    int number_rows = 32;
    int number_cols = 32;

    vector< vector < vector< vector< vector<char> > > > > images(number_labels);

    unsigned char pixel;
    unsigned char label;
    for (uint32_t file = 0; file < image_filenames.size(); file++) {
        ifstream image_file(image_filenames[file].c_str(), ios::in | ios::binary);

        if (!image_file.is_open()) {
            cerr << "Could not open '" << image_filenames[file] << "' for reading." << endl;
            exit(1);
        }

        for (uint32_t i = 0; i < expected_images; i++) {
            image_file.read( (char*)&label, sizeof(char) );
            //cout << "label: " << (int)label << endl;

            vector< vector < vector<char> > > image(3, vector< vector<char> >(32, vector<char> (32) ) );

            for (uint32_t z = 0; z < number_channels; z++) { 
                for (uint32_t y = 0; y < number_cols; y++) {
                    for (uint32_t x = 0; x < number_rows; x++) {
                        image_file.read( (char*)&pixel, sizeof(char) );

                        image[z][y][x] = pixel;
                    }
                }
            }

            images[ (int)label ].push_back(image);

            //char keystroke;
            //cin >> keystroke;
        }

        image_file.close();
    }


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

            for (uint32_t z = 0; z < number_channels; z++) {
                for (int y = 0; y < number_cols; y++) {
                    for (int x = 0; x < number_rows; x++) {
                        pixel = images[i][j][z][y][x];

                        outfile.write( (char*)&pixel, sizeof(char));

                        //cout << " " << (uint32_t)pixel;
                    }
                    //cout << endl;
                }
                //cout << endl;
            }
        }
        cout << "wrote " << images[i].size() << " images." << endl;
    }

    outfile.close();

    return 0;
}

