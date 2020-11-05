import pandas as pd
import numpy as np

# Inputs and outputs
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
# Correct answers to show if we correctly set weights and bias
correct_outputs = [False, False, False, True]  
outputs = []

# Set weights, and bias
weights = [1.0, 0.5]
bias = -1.5

# Generate and check output
for test_input, correct_output in zip(inputs, correct_outputs):
    # Counting linear combination for input
    linear_combination = np.dot(test_input, weights) + bias
    
    # Counting output
    output = int(linear_combination >= 0)
    
    # Check if output is correct
    if output == correct_output:
        is_correct_string = 'Yes' 
    else: 
        is_correct_string = 'No'
        
    # Add everything to generall output
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Printing output
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
print(output_frame.to_string(index=False))
