import matplotlib.pyplot as plt
import numpy as np
import os

# Create a simple image
def create_image():
    # Generate some data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create a plot
    plt.figure()
    plt.plot(x, y, label='sin(x)')
    plt.title("Test Image")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend()

    # Save the plot as an image
    save_image(plt)

# Save the image to the current directory
def save_image(plt):

    # Save the image
    plt.savefig("novel_views/test_image.png", bbox_inches='tight')
    # Close the plot to free up memory
    plt.close()

# Main function
if __name__ == '__main__':
    create_image()