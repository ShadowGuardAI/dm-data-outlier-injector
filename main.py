import argparse
import logging
import pandas as pd
import numpy as np
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_argparse():
    """
    Sets up the argument parser for the command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Injects statistically plausible outlier data points into numerical columns to distort distributions and mask genuine outliers."
    )

    parser.add_argument("input_file", help="Path to the input CSV file.")
    parser.add_argument("output_file", help="Path to the output CSV file.")
    parser.add_argument("column_name", help="Name of the column to inject outliers into.")
    parser.add_argument("num_outliers", type=int, help="Number of outliers to inject.")
    parser.add_argument("std_multiplier", type=float, help="Standard deviation multiplier for outlier generation.")

    return parser

def inject_outliers(df, column_name, num_outliers, std_multiplier):
    """
    Injects outlier data points into the specified column of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to inject outliers into.
        num_outliers (int): The number of outliers to inject.
        std_multiplier (float): The standard deviation multiplier for outlier generation.

    Returns:
        pd.DataFrame: The DataFrame with injected outliers.  Returns None if an error occurs.
    """
    try:
        # Validate inputs
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame.")
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            raise TypeError(f"Column '{column_name}' must be numeric.")
        if num_outliers < 0:
            raise ValueError("Number of outliers must be non-negative.")
        if std_multiplier <= 0:
            raise ValueError("Standard deviation multiplier must be positive.")

        # Calculate mean and standard deviation
        mean = df[column_name].mean()
        std = df[column_name].std()

        # Generate outliers
        outliers = np.random.normal(mean, std * std_multiplier, num_outliers)

        # Append outliers to the column
        df = pd.concat([df, pd.DataFrame({column_name: outliers})], ignore_index=True)

        logging.info(f"Successfully injected {num_outliers} outliers into column '{column_name}'.")
        return df

    except (TypeError, ValueError) as e:
        logging.error(f"Error injecting outliers: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None


def main():
    """
    Main function to execute the outlier injection process.
    """
    parser = setup_argparse()
    args = parser.parse_args()

    try:
        # Read the input CSV file
        df = pd.read_csv(args.input_file)

        # Inject outliers
        df_with_outliers = inject_outliers(df, args.column_name, args.num_outliers, args.std_multiplier)

        if df_with_outliers is not None:
            # Save the DataFrame with outliers to the output CSV file
            df_with_outliers.to_csv(args.output_file, index=False)
            logging.info(f"Outlier injection complete. Output saved to '{args.output_file}'.")
        else:
            logging.error("Outlier injection failed.  See previous error messages for details.")
            sys.exit(1) # Exit with an error code
            
    except FileNotFoundError:
        logging.error(f"Input file '{args.input_file}' not found.")
        sys.exit(1)  # Exit with an error code
    except pd.errors.EmptyDataError:
        logging.error(f"Input file '{args.input_file}' is empty.")
        sys.exit(1)  # Exit with an error code
    except pd.errors.ParserError:
        logging.error(f"Error parsing input file '{args.input_file}'.  Ensure it is a valid CSV.")
        sys.exit(1) # Exit with an error code
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1) # Exit with an error code


if __name__ == "__main__":
    # Example Usage (demonstration - remove in production or use a different mechanism)
    # To run from the command line:
    # python your_script_name.py input.csv output.csv numerical_column 10 3.0

    # Create a dummy input CSV for demonstration
    # import pandas as pd
    # import numpy as np
    # data = {'numerical_column': np.random.normal(10, 2, 100), 'categorical_column': ['A'] * 100}
    # df = pd.DataFrame(data)
    # df.to_csv('input.csv', index=False)
    
    main()