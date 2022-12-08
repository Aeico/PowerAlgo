import threading

# Define the target function for the thread
def run_agent(arg):
    # Do some work and return the result
    result = arg * arg
    return result

# Create the thread
thread = threading.Thread(target=run_agent, args=(5,))

# Start the thread
thread.start()

# Wait for the thread to finish
thread.join()

# Get the result of the thread
result = thread.result

# Print the result of the thread
print(result)  # Output: 25