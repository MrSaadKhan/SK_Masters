import os

def combine_lines(input_folder, output_folder, window_size, slide_length):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)

            with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
                lines = infile.readlines()
                num_lines = len(lines)

                # Combine lines in the specified manner
                for i in range(0, num_lines - window_size + 1, slide_length):
                    combined_line = ''.join(lines[i:i + window_size]).replace('\n', ' ')  # Combine lines
                    outfile.write(combined_line.strip() + '\n')


if __name__ == "__main__":
    input_folder = os.path.join(os.getcwd(), "preprocessed_data", "ungrouped")
    output_folder = os.path.join(os.getcwd(), "preprocessed_data", "grouped")
    
    # Set window size and slide length
    window_size = 10
    slide_length = 1
    
    output_folder = os.path.join(output_folder, f"{window_size}_{slide_length}")

    combine_lines(input_folder, output_folder, window_size, slide_length)
    print("Processing complete. Check the output folder for results.")