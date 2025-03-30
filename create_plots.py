import os
import matplotlib.pyplot as plt

def plot_graphs_embedder(stats_list, vector_list, time_descriptions, memory_descriptions):
    # Create and save the Embeddings Creation Time plot
    plt.figure(figsize=(12, 6))
    plt.grid(True)
    for i, desc in enumerate(time_descriptions):
        if desc == "FastText Training":
            continue  # Skip plotting FastText Training Time
        elif desc == "BERT":
            times = [stats[0][i] for stats in stats_list]
            plt.plot(vector_list, times, marker='x', linestyle='dashed', label='BERT')
        else:
            times = [stats[0][i] for stats in stats_list]
            plt.plot(vector_list, times, marker='o', label='FastText')
    # plt.title('Embeddings Creation Time')
    plt.xlabel('Vector Size')
    plt.ylabel('Time (sec)')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/embeddings_creation_time_plot.svg', format='svg', dpi=300, transparent=True)
    plt.savefig('plots/embeddings_creation_time_plot.png', format='png', dpi=300, transparent=True)
    plt.savefig('plots/embeddings_creation_time_plot.pdf', format='pdf', dpi=300, transparent=True)
    plt.close()

    # Create and save the Embeddings Creation Memory plot
    plt.figure(figsize=(12, 6))
    plt.grid(True)
    for i, desc in enumerate(memory_descriptions):
        if desc == "FastText Training":
            continue  # Skip plotting FastText Training Memory Usage per Flow
        elif desc == "BERT":
            memories = [stats[1][i] for stats in stats_list]
            plt.plot(vector_list, memories, marker='x', linestyle='dashed', label='BERT')
        else:
            memories = [stats[1][i] for stats in stats_list]
            plt.plot(vector_list, memories, marker='o', label='FastText')
    # plt.title('Embeddings Creation Memory Usage')
    plt.xlabel('Vector Size')
    plt.ylabel('Memory (MB)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/embeddings_creation_memory_plot.svg', format='svg', dpi=300, transparent=True)
    plt.savefig('plots/embeddings_creation_memory_plot.png', format='png', dpi=300, transparent=True)
    plt.savefig('plots/embeddings_creation_memory_plot.pdf', format='pdf', dpi=300, transparent=True)
    plt.close()

def plot_fasttext_training(stats_list, vector_list):
    # Create and save the FastText Training Time plot
    plt.figure(figsize=(12, 6))
    plt.grid(True)
    times = [stats[0][0] for stats in stats_list]  # Assuming FastText Training Time is the first in the list
    plt.plot(vector_list, times, marker='o', label='FastText Training')
    # plt.title('FastText Training Time')
    plt.xlabel('Vector Size')
    plt.ylabel('Time (sec)')
    plt.legend()
    plt.tight_layout()  # Add this line to make the margins as thin as possible
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/fasttext_training_time_plot.svg', format='svg', dpi=300, transparent=True)
    plt.savefig('plots/fasttext_training_time_plot.png', format='png', dpi=300, transparent=True)
    plt.savefig('plots/fasttext_training_time_plot.pdf', format='pdf', dpi=300, transparent=True)

    plt.close()

    # Create and save the FastText Training Memory plot
    plt.figure(figsize=(12, 6))
    plt.grid(True)
    memories = [stats[1][0] for stats in stats_list]  # Assuming FastText Training Memory Usage is the first in the list
    plt.plot(vector_list, memories, marker='o', label='FastText Training')
    # plt.title('FastText Training Memory Usage')
    plt.xlabel('Vector Size')
    plt.ylabel('Memory (MB)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/fasttext_training_memory_plot.svg', format='svg', dpi=300, transparent=True)
    plt.savefig('plots/fasttext_training_memory_plot.png', format='png', dpi=300, transparent=True)
    plt.savefig('plots/fasttext_training_memory_plot.pdf', format='pdf', dpi=300, transparent=True)
    plt.close()

def plot_graphs_classifier(stats_list, vector_list, time_descriptions, memory_descriptions, training_length = 1, testing_length = 1):

    if training_length == 1 or testing_length == 1:
        print("Warning: only 1 instance or taking for all flows!")

    # Create and save the Classification Time plot
    plt.figure(figsize=(12, 6))
    plt.grid(True)
    for i, desc in enumerate(time_descriptions):
        times = [stats[0][i] for stats in stats_list]
        times = [time / (testing_length) for time in times]
        if desc == "BERT":  # Change this line to crosses and dotted
            plt.plot(vector_list, times, marker='x', linestyle='dashed', label='BERT')
        else:
            plt.plot(vector_list, times, marker='o', label=desc)
    # plt.title('Embeddings Classification Time')
    plt.xlabel('Vector Size')
    plt.ylabel('Time (sec)')
    plt.legend()
    plt.tight_layout()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/embeddings_classifier_time_plot.svg', format='svg', dpi=300, transparent=True)
    plt.savefig('plots/embeddings_classifier_time_plot.png', format='png', dpi=300, transparent=True)
    plt.savefig('plots/embeddings_classifier_time_plot.pdf', format='pdf', dpi=300, transparent=True)
    plt.close()

    # Create and save the Classification Memory plot
    plt.figure(figsize=(12, 6))
    plt.grid(True)
    for i, desc in enumerate(memory_descriptions):
        memories = [stats[1][i] for stats in stats_list]
        # memories = [memories / (testing_length) for time in times]
        memories = [memory / testing_length for memory in memories]

        if desc == "BERT":  # Change this line to crosses and dotted
            plt.plot(vector_list, memories, marker='x', linestyle='dashed', label='BERT')
        else:
            plt.plot(vector_list, memories, marker='o', label=desc)
    # plt.title('Embeddings Classification Memory Usage')
    plt.xlabel('Vector Size')
    plt.ylabel('Memory (MB)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/embeddings_classifier_memory_plot.svg', format='svg', dpi=300, transparent=True)
    plt.savefig('plots/embeddings_classifier_memory_plot.png', format='png', dpi=300, transparent=True)
    plt.savefig('plots/embeddings_classifier_memory_plot.pdf', format='pdf', dpi=300, transparent=True)
    plt.close()

# if __name__ == "__main__":
#     # Example data for plot_graphs_embedder
#     stats_list_embedder = [
#         [(10.2, 15.5, 20.1), (100, 120, 130)],
#         [(8.5, 12.9, 18.7), (90, 110, 115)],
#         [(7.1, 10.8, 15.2), (80, 100, 105)],
#         [(6.2, 9.5, 13.8), (75, 95, 100)],
#         [(5.5, 8.7, 12.1), (70, 90, 95)],
#         [(4.8, 7.9, 10.5), (65, 85, 90)],
#         [(3.9, 6.5, 9.1), (60, 80, 85)],
#         [(3.2, 5.8, 7.8), (55, 75, 80)]
#     ]
#     vector_list = [768, 512, 256, 128, 64, 32, 15, 5]
#     time_descriptions_embedder = [
#         "FastText Training Time",
#         "FastText Embeddings Creation Time per Flow",
#         "BERT Embeddings Creation Time per Flow"
#     ]
#     memory_descriptions_embedder = [
#         "FastText Training Memory Usage per Flow",
#         "FastText Embeddings Creation Memory Usage per Flow",
#         "BERT Embeddings Creation Memory Usage per Flow"
#     ]

#     plot_graphs_embedder(stats_list_embedder, vector_list, time_descriptions_embedder, memory_descriptions_embedder)
#     plot_fasttext_training(stats_list_embedder, vector_list)

#     # Example data for plot_graphs_classifier
#     stats_list_classifier = [
#         [(10.2, 15.5), (100, 120)],
#         [(8.5, 12.9), (90, 110)],
#         [(7.1, 10.8), (80, 100)],
#         [(6.2, 9.5), (75, 95)],
#         [(5.5, 8.7), (70, 90)],
#         [(4.8, 7.9), (65, 85)],
#         [(3.9, 6.5), (60, 80)],
#         [(3.2, 5.8), (55, 75)]
#     ]
#     time_descriptions_classifier = [
#         "FastText",
#         "BERT"
#     ]
#     memory_descriptions_classifier = [
#         "FastText",
#         "BERT"
#     ]

#     plot_graphs_classifier(stats_list_classifier, vector_list, time_descriptions_classifier, memory_descriptions_classifier)
