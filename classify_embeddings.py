import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bars
import time
# from memory_profiler import memory_usage
import gc
import create_plots
import joblib
import tensorflow as tf  # Added for neural network functionality

# Mapping device names to their indices
def map_device_name(file_paths):
    device_names = []
    for fp in file_paths:
        # Extract device name from the file name before ".json"
        device_name = os.path.basename(fp).split('.json')[0].replace('_', ' ')
        device_name = ' '.join(word.capitalize() for word in device_name.split())
        device_names.append(device_name)
        
    unique_devices = sorted(set(device_names))
    device_to_index = {device: idx for idx, device in enumerate(unique_devices)}
    return device_to_index

def plot_device_names(input_list):
    # Dictionary mapping device names to plot names
    input_output_map = {
        "Amazon Echo Gen2"                      : "Amazon Echo Gen2",
        "Au Network Camera"                     : "Network Camera",
        "Au Wireless Adapter"                   : "Wireless Adapter",
        "Bitfinder Awair Breathe Easy"          : "Bitfinder Smart Air Monitor",
        "Candy House Sesami Wi-fi Access Point" : "Candy House Wi-Fi AP",
        "Google Home Gen1"                      : "Google Home Gen1",
        "I-o Data Qwatch"                       : "IO Data Camera",
        "Irobot Roomba"                         : "iRobot Roomba",
        "Jvc Kenwood Cu-hb1"                    : "JVC Smart Home Hub",
        "Jvc Kenwood Hdtv Ip Camera"            : "JVC Camera",
        "Line Clova Wave"                       : "Line Smart Speaker",
        "Link Japan Eremote"                    : "Link eRemote",
        "Mouse Computer Room Hub"               : "Mouse Computer Room Hub",
        "Nature Remo"                           : "Nature Smart Remote",
        "Panasonic Doorphone"                   : "Panasonic Doorphone",
        "Philips Hue Bridge"                    : "Philips Hue Light",
        "Planex Camera One Shot!"               : "Planex Camera",
        "Planex Smacam Outdoor"                 : "Planex Outdoor Camera",
        "Planex Smacam Pantilt"                 : "Planex PanTilt Camera",
        "Powerelectric Wi-fi Plug"              : "PowerElectric Wi-Fi Plug",
        "Qrio Hub"                              : "Qrio Hub",
        "Sony Bravia"                           : "Sony Bravia",
        "Sony Network Camera"                   : "Sony Network Camera",
        "Sony Smart Speaker"                    : "Sony Smart Speaker",
        "Xiaomi Mijia Led"                      : "Xiaomi Mijia LED"
    }

    output_list = []
    for input_string in input_list:
        output = input_output_map.get(input_string)
        if output is None:
            print(f"Input {input_string} not found")
            output_list.append("")
        else:
            output_list.append(output)
    
    return output_list

def classify_embeddings_random_forest(folder_path, output_name, vector_size):
    
    def load_embeddings(file_path):
        embeddings = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc=f'Loading embeddings from {os.path.basename(file_path)}', unit=' vectors'):
                vector = np.array([float(x) for x in line.strip().split()])
                embeddings.append(vector)
        return embeddings

    # List of file paths in the folder
    file_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith('.txt')]

    # Map device names to indices
    device_to_index = map_device_name(file_paths)

    # Load embeddings and labels
    all_embeddings = []
    all_labels = []
    for file_path in sorted(file_paths):  # Sorted to ensure seen and unseen pairs are together
        # Extract device name from the file name before ".json"
        device_name = os.path.basename(file_path).split('.json')[0].replace('_', ' ')
        device_name = ' '.join(word.capitalize() for word in device_name.split())
        
        if device_name not in device_to_index:
            print(f"Device name '{device_name}' not found in device_to_index dictionary.")
            continue
        
        device_index = device_to_index[device_name]
        device_embeddings = load_embeddings(file_path)
        labels = [device_index] * len(device_embeddings)
        all_embeddings.extend(device_embeddings)
        all_labels.extend(labels)

    # Convert to numpy arrays
    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(all_embeddings, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
    
    training_length = len(X_train)
    testing_length  = len(X_test)

    # Initialize and train the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=500, random_state=42)
    clf.fit(X_train, y_train)
    print('RF Training completed.')

    # Classify with progress bar
    y_pred = []
    for batch in tqdm(np.array_split(X_test, 10), desc='RF Classifying', unit=' batches'):
        y_pred.extend(clf.predict(batch))
    y_pred = np.array(y_pred)

    print(f"Evaluation of RF classifier at a vector size of {vector_size}")
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"RF F1 Score for {vector_size}: {f1} \n (Folder: {folder_path})")

    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]  # Convert to percentage
    accuracy = np.mean(np.diag(conf_matrix_percent))

    device_names = sorted(device_to_index, key=device_to_index.get)
    device_names = plot_device_names(device_names)

    # Dynamically compute font size based on the confusion matrix dimensions
    matrix_size = conf_matrix_percent.shape[0]
    font_size = 100 / matrix_size  # Tweak scaling factor as needed
    print(f"Font size = {font_size}")
    
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_percent, display_labels=device_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(X_test) > 10:
        fig, ax = plt.subplots(figsize=(16, 12))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f", text_kw={'fontsize': font_size})
    
    # Set axis labels and tick parameters using the dynamic font size
    ax.set_xlabel('Predicted Label', fontsize=font_size)
    ax.set_ylabel('True Label', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.set_xticklabels(device_names, rotation=90, fontsize=font_size)
    ax.set_yticklabels(device_names, fontsize=font_size)
    
    plt.tight_layout()
    ax.figure.savefig(f'plots/{output_name}_confusion_matrix_rf_{vector_size}.png', dpi=300, transparent=True)
    ax.figure.savefig(f'plots/{output_name}_confusion_matrix_rf_{vector_size}.svg', dpi=300, transparent=True)
    ax.figure.savefig(f'plots/{output_name}_confusion_matrix_rf_{vector_size}.pdf', dpi=300, transparent=True)

    folder_path_rf = './rfmodels/'
    os.makedirs(folder_path_rf, exist_ok=True)
    model_file = os.path.join(folder_path_rf, f'{output_name}_random_forest_model.pkl')
    joblib.dump(clf, model_file)

    file_size_bytes = os.path.getsize(model_file)
    file_size = file_size_bytes / (1024 * 1024)

    return accuracy, file_size, training_length, testing_length

def classify_embeddings_nn(folder_path, output_name, vector_size):
    def load_embeddings(file_path):
        embeddings = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc=f'Loading embeddings from {os.path.basename(file_path)}', unit=' vectors'):
                vector = np.array([float(x) for x in line.strip().split()])
                embeddings.append(vector)
        return embeddings

    # List of file paths in the folder
    file_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith('.txt')]

    # Map device names to indices
    device_to_index = map_device_name(file_paths)

    # Load embeddings and labels
    all_embeddings = []
    all_labels = []
    for file_path in sorted(file_paths):
        device_name = os.path.basename(file_path).split('.json')[0].replace('_', ' ')
        device_name = ' '.join(word.capitalize() for word in device_name.split())
        
        if device_name not in device_to_index:
            print(f"Device name '{device_name}' not found in device_to_index dictionary.")
            continue
        
        device_index = device_to_index[device_name]
        device_embeddings = load_embeddings(file_path)
        labels = [device_index] * len(device_embeddings)
        all_embeddings.extend(device_embeddings)
        all_labels.extend(labels)

    # Convert to numpy arrays
    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(all_embeddings, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
    training_length = len(X_train)
    testing_length = len(X_test)
    
    n_classes = len(device_to_index)
    # Build a simple feed-forward neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(vector_size,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0)
    
    # Predict on test set
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print(f"Evaluation of NN classifier at a vector size of {vector_size}")
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"NN F1 Score for {vector_size}: {f1} \n (Folder: {folder_path})")
    
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    device_names = sorted(device_to_index, key=device_to_index.get)
    device_names = plot_device_names(device_names)

    # Dynamically compute font size based on the confusion matrix dimensions
    matrix_size = conf_matrix_percent.shape[0]
    font_size = 100 / matrix_size  # Tweak scaling factor as needed
    print(f"Font size = {font_size}")
    
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_percent, display_labels=device_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(X_test) > 10:
        fig, ax = plt.subplots(figsize=(16, 12))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f", text_kw={'fontsize': font_size})
    
    # Set axis labels and tick parameters using the dynamic font size
    ax.set_xlabel('Predicted Label', fontsize=font_size)
    ax.set_ylabel('True Label', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.set_xticklabels(device_names, rotation=90, fontsize=font_size)
    ax.set_yticklabels(device_names, fontsize=font_size)
    
    plt.tight_layout()
    ax.figure.savefig(f'plots/{output_name}_confusion_matrix_nn_{vector_size}.png', dpi=300, transparent=True)
    ax.figure.savefig(f'plots/{output_name}_confusion_matrix_nn_{vector_size}.svg', dpi=300, transparent=True)
    ax.figure.savefig(f'plots/{output_name}_confusion_matrix_nn_{vector_size}.pdf', dpi=300, transparent=True)
    
    # Save the NN model
    folder_path_nn = './nnmodels/'
    os.makedirs(folder_path_nn, exist_ok=True)
    model_file = os.path.join(folder_path_nn, f'{output_name}_nn_model_{vector_size}.h5')
    model.save(model_file)
    file_size_bytes = os.path.getsize(model_file)
    file_size = file_size_bytes / (1024 * 1024)

    return f1, file_size, training_length, testing_length

def plot_accuracy_vs_vector_size(data):
    bert_data = [item for item in data if 'bert_embeddings' in item[1] and 'nn' not in item[1]]
    fasttext_data = [item for item in data if 'fast_text_embeddings' in item[1] and 'nn' not in item[1]]
    nn_data = [item for item in data if 'nn' in item[1]]

    plt.figure(figsize=(12, 6))
    plt.plot([item[0] for item in bert_data], [item[2] for item in bert_data], marker='x', linestyle='dashed', label='BERT (RF)')
    plt.plot([item[0] for item in fasttext_data], [item[2] for item in fasttext_data], marker='o', label='FastText (RF)')
    plt.plot([item[0] for item in nn_data], [item[2] for item in nn_data], marker='s', label='NN')
    plt.xlabel('Vector Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.savefig('plots/classifier_accuracy.png', format='png', dpi=300, transparent=True)
    plt.savefig('plots/classifier_accuracy.svg', format='svg', dpi=300, transparent=True)
    plt.savefig('plots/classifier_accuracy.pdf', format='pdf', dpi=300, transparent=True)

def main(vector_list, device_range, vector_path, group_option, window_size, slide_length):
    file_path = vector_path
    stats_list = []

    time_descriptions = [
        "FastText",
        "BERT"
    ]
    memory_descriptions = [
        "FastText",
        "BERT"
    ]

    if group_option == 0:
        group_option = "Ungrouped"
    else:
        group_option = "Grouped"

    embed_options = ["bert_embeddings", "bert_embeddings_finetuned", "fast_text_embeddings", "gpt2_embeddings"]  # Embedding options
    more_options = ["BERT", "BERT", "FastText", "GPT2"]

    accuracy_list = []  # List to store accuracies

    print(vector_list)
    for vector_size in vector_list:
        print(f"Classifying embeddings at vector size: {vector_size}")

        bert_embeddings_classification_time = 0
        bert_embeddings_classification_mem_usage = 0
        fast_text_embeddings_classification_time = 0
        fast_text_embeddings_classification_mem_usage = 0

        for option in embed_options:
            print(f"option is: {option}")
            embed_name = f"{option}"
            folder_path = os.path.join(file_path, str(vector_size), more_options[embed_options.index(option)], group_option, f"{window_size}_{slide_length}", embed_name)
            print(f"Folder to analyze: {folder_path}")
            memory = 0

            gc.collect()
            start_time = time.time()

            if os.path.exists(folder_path):
                # Random Forest Classification
                rf_accuracy, memory, training_length, testing_length = classify_embeddings_random_forest(folder_path, embed_name, vector_size)
                accuracy_list.append((vector_size, option + '_rf', rf_accuracy))
                print(f"RF Accuracy for {embed_name}: {rf_accuracy}")

                if option.startswith("bert_embeddings"):
                    bert_embeddings_classification_time = time.time() - start_time
                    bert_embeddings_classification_mem_usage = memory

                if option.startswith("fast_text_embeddings"):
                    fast_text_embeddings_classification_time = time.time() - start_time
                    fast_text_embeddings_classification_mem_usage = memory

                # Neural Network Classification
                nn_accuracy, nn_model_size, nn_train_length, nn_test_length = classify_embeddings_nn(folder_path, embed_name, vector_size)
                accuracy_list.append((vector_size, option + '_nn', nn_accuracy))
                print(f"NN Accuracy for {embed_name}: {nn_accuracy}")

            else:
                print(f"{embed_name} does not exist!")
                print(f"Expected path: {folder_path}")

            print(f"Time taken: {time.time() - start_time:.2f} seconds")

        stats_list.append((
            (fast_text_embeddings_classification_time, bert_embeddings_classification_time),
            (fast_text_embeddings_classification_mem_usage, bert_embeddings_classification_mem_usage)
        ))
    print(stats_list)
    plot_accuracy_vs_vector_size(accuracy_list)
    create_plots.plot_graphs_classifier(stats_list, vector_list, time_descriptions, memory_descriptions, training_length, testing_length)

def main_ext(vector_list, device_low, device_high, group_option, time_group, num2word_option, window_group, window_size, slide_length):
    device_range = f"{device_low}-{device_high}"
    vector_path = os.path.join(os.getcwd(), device_range)
    print(vector_path)
    print(vector_list)
    main(vector_list, device_range, vector_path, group_option, window_size, slide_length)
    print("Complete :)")
