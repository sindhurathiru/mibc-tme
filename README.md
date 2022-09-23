# Evaluation of MIBC Patients' TME
<i>A preliminary analysis demonstrating the application of machine learning in an automated pipeline to study the tumour microenvironment in matched biopsy and cystectomy tissue samples.</i>

The model used is an autoencoder for an unsupervised approach in order to obtain a qualitative comparison of the distribution of cell types between biopsy and cystectomy tissue samples. An autoencoder was chosen since it allows for visualization of the data in a lower dimensional space through use of the encoding layer, referred to herein as the latent space. The latent space ultimately provides a more informative understanding of the dataset in its entirety.

Found that the proteomic profiles of biopsy tissues seem to follow a very siimilar distribution as the matched cystectomy tissues. The similar expression profiles indicate the possibility for pathologists to deduce the same information obtained from a cystectomy sample, but from a biopsy sample instead. In doing so, clinicians would be able to administer proper treatment to the patient without having to wait for a cystectomy.

## Files
<b>cysVtur.py</b> contains code to complete this analysis using a deep autoencoder. Each layer of the autoencoder uses a ReLU activation function, with the final encoding and decoding layer using a linear and sigmoid activation function respectively. The network was trained using the Adam optimization function with a learning rate of 5e<sup>-5</sup>. The mean squared error was calculated and back-propagated throughout training of the network. A latent space of dimension size 2 was used in order for optimal visualization of data results.
