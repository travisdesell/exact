#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>

#include "common/arguments.hxx"
#include "common/version.hxx"

#include "image_tools/image_set.hxx"

#include "cnn/cnn_genome.hxx"

#include "stdint.h"

/**
 *  *  Includes required for BOINC
 *   */

#ifdef _WIN32
    #include "boinc_win.h"
    #include "str_util.h"
#endif

#include "util.h"
#include "filesys.h"
#include "boinc_api.h"
#include "mfile.h"

#include "diagnostics.h"

using namespace std;

vector<string> arguments;


int progress_function(float progress) {
    boinc_checkpoint_completed();
    return boinc_fraction_done(progress);
}

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

	cerr << "arguments:" << endl;
	for (int32_t i = 0; i < argc; i++) {
		cerr << "\t'" << argv[i] << "'" << endl;
	}

    cerr << "converting arguments to vector" << endl;
    arguments = vector<string>(argv, argv + argc);

    string training_filename;
    string testing_filename;
    string validation_filename;
    string genome_filename;
    string output_filename;
    string checkpoint_filename;

    get_argument(arguments, "--training_file", true, training_filename);
    get_argument(arguments, "--validation_file", true, validation_filename);
    get_argument(arguments, "--testing_file", true, testing_filename);
    get_argument(arguments, "--genome_file", true, genome_filename);
    get_argument(arguments, "--output_file", true, output_filename);
    get_argument(arguments, "--checkpoint_file", true, checkpoint_filename);

    training_filename = get_boinc_filename(training_filename);
    validation_filename = get_boinc_filename(validation_filename);
    testing_filename = get_boinc_filename(testing_filename);
    genome_filename = get_boinc_filename(genome_filename);
    output_filename = get_boinc_filename(output_filename);
    checkpoint_filename = get_boinc_filename(checkpoint_filename);

    cerr << "boincified training filename: '" << training_filename << "'" << endl;
    cerr << "boincified validation filename: '" << validation_filename << "'" << endl;
    cerr << "boincified testing filename: '" << testing_filename << "'" << endl;
    cerr << "boincified genome filename: '" << genome_filename << "'" << endl;
    cerr << "boincified output filename: '" << output_filename << "'" << endl;
    cerr << "boincified checkpoint filename: '" << checkpoint_filename << "'" << endl;

    cerr << "loading genome" << endl;

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
    cerr << "parsed input file" << endl;

    if (!genome->sanity_check(SANITY_CHECK_AFTER_GENERATION)) {
        cerr << "ERROR! genome failed sanity check! This should never happen!" << endl;

        boinc_finish(1);
        exit(1);
    }

    if (genome->get_version_str().compare(EXACT_VERSION_STR) != 0) {
        cerr << "ERROR: exact application with version '" << EXACT_VERSION_STR << "' trying to process workunit with incompatible input version: '" << genome->get_version_str() << "'" << endl;
        boinc_finish(1);
        exit(1);
    }

    cerr << "loaded genome" << endl;

    cerr << "loading images" << endl;

    Images training_images(training_filename, genome->get_padding());
    Images validation_images(validation_filename, genome->get_padding(), training_images.get_average(), training_images.get_std_dev());
    Images testing_images(testing_filename, genome->get_padding(), training_images.get_average(), training_images.get_std_dev());

    if (!training_images.loaded_correctly()) {
        cerr << "ERROR: had error loading training images" << endl;
        boinc_finish(1);
        exit(1);
    } else if (!validation_images.loaded_correctly()) {
        cerr << "ERROR: had error loading validation images" << endl;
        boinc_finish(1);
        exit(1);
    } else if (!testing_images.loaded_correctly()) {
        cerr << "ERROR: had error loading testing images" << endl;
        boinc_finish(1);
        exit(1);
    }

    cerr << "loaded images" << endl;


    //genome->print_graphviz(cout);

    genome->set_progress_function(progress_function);

    genome->set_checkpoint_filename(checkpoint_filename);
    genome->set_output_filename(output_filename);

    cerr << "starting backpropagation!" << endl;
    genome->stochastic_backpropagation(training_images, validation_images);
    genome->evaluate_test(testing_images);
    cerr << "backpropagation finished successfully!" << endl;

    genome->write_to_file(output_filename);

    boinc_finish(0);

    return 0;
}

#ifdef _WIN32

#if defined(_MSC_VER) && (_MSC_VER >= 1400)
void AppInvalidParameterHandler(const wchar_t* expression, const wchar_t* function, const wchar_t* file, unsigned int line, uintptr_t pReserved) {
	DebugBreak();
}
#endif


int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR Args, int WinMode){
    LPSTR command_line;
    char* argv[100];
    int argc;

#if defined(_MSC_VER) && (_MSC_VER >= 1400)
	_set_invalid_parameter_handler(AppInvalidParameterHandler);
#endif

    command_line = GetCommandLine();
    argc = parse_command_line( command_line, argv );
    return main(argc, argv);
}
#endif

