def extract_req_cols(columns_list: list, suffix: str) -> list:
    """
    Extract required columns based on the suffix in the column names. e.g., NSDN_RETURN

    @param columns_list list List of all column names.
    @param suffix str 
        Suffix str to extract its respective columns. e.g., VOL_CHANGE, RETURN
    
    @return required_cols List of required column names for the given suffix
    """
    required_cols = [col for col in columns_list if suffix in col]
    return required_cols