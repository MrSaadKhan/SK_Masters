import psutil
import time

def print_memory_usage():
    while True:
        mem = psutil.virtual_memory()
        print(f"Total Memory: {mem.total / (1024 ** 2):.2f} MB")
        print(f"Available Memory: {mem.available / (1024 ** 2):.2f} MB")
        print(f"Used Memory: {mem.used / (1024 ** 2):.2f} MB")
        print(f"Memory Percentage: {mem.percent}%")
        print("-" * 30)
        time.sleep(1)  # Print every 1 second

print_memory_usage()
