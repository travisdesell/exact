#include "common/arguments.hxx"
#include "common/log.hxx"

#include "oracle.hxx"
#include "portfolio_oracle.hxx"

Oracle* Oracle::initialize_from_arguments(const vector<string> &arguments) {

    string oracle_type;
    get_argument(arguments, "--oracle_type", true, oracle_type);

    if (oracle_type == "portfolio") {
        return new PortfolioOracle(arguments);

    } else {
        Log::fatal("unknown oracle type '%s', cannot evaluate oracle.\n\n", oracle_type.c_str());
        exit(1);
    }

}
