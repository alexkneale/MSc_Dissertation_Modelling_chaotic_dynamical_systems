import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for the Gaussian components
weights = [0.2, 0.3, 0.5]
means = [1, 3, 5]
variances = [2, 3, 1]

# Create a function to calculate the PDF of a Gaussian distribution
def gaussian_pdf(x, mean, variance):
    return norm.pdf(x, loc=mean, scale=np.sqrt(variance))

# Create an array of x values
x = np.linspace(-5, 10, 1000)

# Calculate the MDN PDF by summing the weighted PDFs of the Gaussian components
mdn_pdf = np.zeros_like(x)
for weight, mean, variance in zip(weights, means, variances):
    mdn_pdf += weight * gaussian_pdf(x, mean, variance)

# Plot the MDN PDF
plt.plot(x, mdn_pdf, label='MDN PDF', color='blue')

# Plot individual Gaussian components for reference
for i, (weight, mean, variance) in enumerate(zip(weights, means, variances)):
    component_pdf = weight * gaussian_pdf(x, mean, variance)
    plt.plot(x, component_pdf, label=r'Component {i+1}: {weight}*$N$({mean},{variance})', linestyle='--')

# Set plot labels and title
plt.xlabel(r'$x$')
plt.ylabel('Probability Density')
plt.title('PDF of Mixture Density Network')
plt.legend()

# Save the plot as a PDF file
plt.savefig('mdn_pdf.pdf')

# Show the plot (optional, you can comment this out if you don't need to display it)
plt.show()