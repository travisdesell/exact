#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include <boost/filesystem.hpp>
using boost::filesystem::create_directories;
using boost::filesystem::directory_iterator;
using boost::filesystem::is_directory;

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

//from UNDVC_COMMON
#include "vector_io.hxx"
#include "arguments.hxx"

//from TAO
#include "neural_networks/convolutional_neural_network.hxx"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);


    int rowscols;
    get_argument(arguments, "--rowscols", true, rowscols);

    int img_reduction = 128 / rowscols;

    string conv_net_file, input_image, output_file;
    get_argument(arguments, "--conv_net", true, conv_net_file);
    get_argument(arguments, "--image", true, input_image);
    get_argument(arguments, "--output_file", true, output_file);

    vector<int> conv_sizes;
    vector<int> max_pool_sizes;
    get_argument_vector(arguments, "--convolutional_sizes", true, conv_sizes);
    get_argument_vector(arguments, "--max_pool_sizes", true, max_pool_sizes);

    int n_classes;
    get_argument(arguments, "--n_classes", true, n_classes);

    if (conv_sizes.size() != max_pool_sizes.size()) {
        cerr << "ERROR: convolutional_sizes.size() [" << conv_sizes.size() << "] != max_pool_sizes.size() [" << max_pool_sizes.size() << "]" << endl;
        exit(1);
    }
    vector< pair<int,int> > layers;
    for (int i = 0; i < conv_sizes.size(); i++) {
        layers.push_back(pair<int, int>(conv_sizes[i], max_pool_sizes[i]));
    }

    int fc_size;
    get_argument(arguments, "--fully_connected_size", true, fc_size);


    cout << "opening conv net file: " << conv_net_file << endl;
    ifstream infile(conv_net_file.c_str());

    string line;
    getline(infile, line);
    getline(infile, line);

    vector<double> weights;
    string_to_vector(line, weights);

    cout << "weights vector size: " << weights.size() << endl;

    infile.close();

    cout << "opening image: " << input_image << endl;
    Mat src = imread( input_image.c_str(), CV_LOAD_IMAGE_COLOR);

    cout << "source image size:       " << src.cols << ", rows: " << src.rows << endl;

    Size size(src.cols / img_reduction, src.rows / img_reduction); /*width, height*/

    Mat resized_image;
    resize(src, resized_image, size);
    cout << "resized image size:       " << resized_image.cols << ", rows: " << resized_image.rows << endl;

    imshow("resized image", resized_image);
    imwrite("resized_image.png", resized_image);

    vector< vector< vector<char> > > images(n_classes);    //can ignore this

    bool quiet = false;
    ConvolutionalNeuralNetwork *conv_nn = new ConvolutionalNeuralNetwork(rowscols, rowscols, true, quiet, images, layers, fc_size);


    conv_nn->set_weights(weights);

    Mat classified_image;
    if (n_classes == 2) {
        classified_image = Mat::zeros( resized_image.rows - rowscols, resized_image.cols - rowscols, CV_8UC1 );
    } else if (n_classes == 3) {
        classified_image = Mat::zeros( resized_image.rows - rowscols, resized_image.cols - rowscols, CV_8UC3 );
    } else {
        cerr << "not sure what to do with " << n_classes << " classes." << endl;
        exit(1);
    }


    vector<char> image;
    for (int i = 0; i < resized_image.rows; i++) {
        for (int j = 0; j < resized_image.cols; j++) {
            Vec3b pixel = resized_image.at<Vec3b>(i, j);
            image.push_back(pixel[0]);
            image.push_back(pixel[1]);
            image.push_back(pixel[2]);
        }
    }

    /*
    conv_nn->initialize_opencl();

    vector<float> output = conv_nn->apply_to_image_opencl(image, resized_image.rows, resized_image.cols, 0);
    int current = 0;
    for (int i = 0; i < resized_image.rows - rowscols; i++) {
        for (int j = 0; j < resized_image.cols - rowscols; j++) {
            classified_image.at<uchar>(i, j) = (short)(output[current] * 256.0);
            current++;
        }
    }

    conv_nn->deinitialize_opencl();
    */

    for (int i = 0; i < resized_image.rows - rowscols; i++) {
        for (int j = 0; j < resized_image.cols - rowscols; j++) {
            vector<char> current_image;

            for (int k = 0; k < rowscols; k++) {
                for (int l = 0; l < rowscols; l++) {
                    Vec3b pixel = resized_image.at<Vec3b>(i + k, j + l);
                    current_image.push_back(pixel[0]);
                    current_image.push_back(pixel[1]);
                    current_image.push_back(pixel[2]);
                }
            }

            conv_nn->evaluate(current_image, 0);

            //cout << "at " << i << ", " << j << ": " << conv_nn->get_output_class(0) << ", " << conv_nn->get_output_class(1) << ", " << conv_nn->get_output_class(2) << endl;

            if (n_classes == 2) {
                short b = (short)(conv_nn->get_output_class(0) * 255.0);
                classified_image.at<uchar>(i, j) = b;
            } else {
                short b = (short)(conv_nn->get_output_class(0) * 255.0);
                short g = 0;
                short r = (short)(conv_nn->get_output_class(1) * 255.0);
                classified_image.at<Vec3b>(i, j) = Vec3b(r, g, b);
            }
        }

        cout << i << " / " << (resized_image.rows - rowscols) << endl;
    }

    imshow("classified image", classified_image);
    imwrite(output_file.c_str(), classified_image);

    waitKey(0);

    return 0;
}

