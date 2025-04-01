import matplotlib.pyplot as plt

# Version names and corresponding accuracy
versions = ['L1 distance 1NN', 'L2 distance 1NN', 'L1 distance 10NN','Basic SVN', 'Softmax']
accuracies = [27.3, 25.4, 29.3, 32.0, 42.0]  # Example accuracy values (replace with actual)

# Create the bar chart
plt.figure(figsize=(10, 8))
plt.bar(versions, accuracies)

# Add labels and title
plt.xlabel('Approach')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
#plt.xticks(rotation=25, ha='right')

plt.tight_layout()

# Display the chart
plt.show()
