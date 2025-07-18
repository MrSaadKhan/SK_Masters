import main_create_all_embeddings
import classify_embeddings
import sys
import os
import traceback
import special

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Define a helper function to redirect output to a file
def redirect_output_to_file(file_path):
    # Open the file for writing
    sys.stdout = open(file_path, 'w')

# Define a helper function to reset output back to the console
def reset_output():
    sys.stdout.close()
    sys.stdout = sys.__stdout__

vector_list = [768]#[512]
device_low = 0
device_high = [5]#[2, 5, 10, 15, 20, 22]

group_option = 0
time_group = 0
num2word_option = 0  # Unlikely to be implemented

window_group = 1
window_size = 10
slide_length = 1

subject = "Code Done!"
body = "The code has been executed successfully."

try:
    for device_high_option in device_high:

        # Redirect output to file for main_create_all_embeddings
        redirect_output_to_file(f"Output1-{device_low}-{device_high_option}.txt")
        main_create_all_embeddings.main_ext(vector_list, device_low, device_high_option, group_option, time_group, num2word_option, window_group, window_size, slide_length)
        reset_output()  # Reset output back to the console


        # Redirect output to file for classify_embeddings
        redirect_output_to_file(f"output3-{device_low}{device_high_option}.txt")
        classify_embeddings.main_ext(vector_list, device_low, device_high_option, group_option, time_group, num2word_option, window_group, window_size, slide_length)
        reset_output()  # Reset output back to the console

        os.rename(os.path.join(os.getcwd(), "plots"), os.path.join(os.getcwd(), f"plots_{device_high_option}"))

    print("All scripts executed successfully and outputs saved to files.")
    special.send_test_email(subject, body)

except Exception as e:
    error_subject = "Code Failed!"
    tb_str = traceback.format_exc()  # Get the full traceback as a string
    error_body = f"The code encountered an error:\n{str(e)}\n\n{tb_str}"
    special.send_test_email(error_subject, error_body)