#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>

//from undvc_common
#include "../undvc_common/arguments.hxx"

#include "image_tools/image_set.hxx"

#include "strategy/cnn_genome.hxx"

/**
 *  *  Includes required for BOINC
 *   */
#ifdef _BOINC_
#ifdef _WIN32
    #include "boinc_win.h"
    #include "str_util.h"
#endif

    #include "diagnostics.h"
    #include "util.h"
    #include "filesys.h"
    #include "boinc_api.h"
    #include "mfile.h"
#endif

using namespace std;

vector<string> arguments;

string get_boinc_filename(string filename) {
    string input_path;

    int retval = boinc_resolve_filename_s(filename.c_str(), input_path);
    if (retval) {
        cerr << "APP: error reading input file (resolving checkpoint file name)" << endl;
        boinc_finish(1);
        exit(1);
    }   

    return input_path;
}



int main(int argc, char** argv) {
    cerr << "arguments:" << endl;
    for (uint32_t i = 0; i < argc; i++) {
        cerr << "\t'" << argv[i] << "'" << endl;
    }

#ifdef _BOINC_
    int retval = 0;
    #ifdef BOINC_APP_GRAPHICS
        #if defined(_WIN32) || defined(__APPLE)
            retval = boinc_init_graphics(worker);
        #else
            retval = boinc_init_graphics(worker, argv[0]);
        #endif
    #else
        retval = boinc_init();
    #endif
    if (retval) exit(retval);
#endif

    cerr << "converting arguments to vector" << endl;
    arguments = vector<string>(argv, argv + argc);

    string binary_samples_filename;
    string genome_filename;
    string output_filename;
    string checkpoint_filename;

    get_argument(arguments, "--samples_file", true, binary_samples_filename);
    get_argument(arguments, "--genome_file", true, genome_filename);
    get_argument(arguments, "--output_file", true, output_filename);
    get_argument(arguments, "--checkpoint_file", true, checkpoint_filename);

    binary_samples_filename = get_boinc_filename(binary_samples_filename);
    genome_filename = get_boinc_filename(genome_filename);
    output_filename = get_boinc_filename(output_filename);
    checkpoint_filename = get_boinc_filename(checkpoint_filename);

    cerr << "boincified samples filename: '" << binary_samples_filename << "'" << endl;
    cerr << "boincified genome filename: '" << genome_filename << "'" << endl;
    cerr << "boincified output filename: '" << output_filename << "'" << endl;
    cerr << "boincified checkpoint filename: '" << checkpoint_filename << "'" << endl;

    cerr << "parsed arguments, loading images" << endl;

    Images images(binary_samples_filename);

    cerr << "loaded images" << endl;

    CNN_Genome *genome = NULL;

    ifstream infile(checkpoint_filename);
    if (infile) {
        //start from the checkpoint if it exists
        cerr << "starting from checkpoint file: '" << checkpoint_filename << "'" << endl;

        genome = new CNN_Genome(checkpoint_filename, true);
    } else {
        //start from the input genome file otherwise
        cerr << "starting from input file: '" << genome_filename << "'" << endl;
        genome = new CNN_Genome(genome_filename, false);
        //genome->set_to_best();
    }
    cerr << "parsed intput file" << endl;

    //genome->print_graphviz(cout);

    genome->set_progress_function(boinc_fraction_done);

    genome->set_checkpoint_filename(checkpoint_filename);
    genome->set_output_filename(output_filename);

    cerr << "starting backpropagation!" << endl;
    genome->stochastic_backpropagation(images);

#ifdef _BOINC_
    boinc_finish(0);
#endif

    return 0;
}

#ifdef _WIN32
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR Args, int WinMode){
    LPSTR command_line;
    char* argv[100];
    int argc;

    command_line = GetCommandLine();
    argc = parse_command_line( command_line, argv );
    return main(argc, argv);
}
#endif

