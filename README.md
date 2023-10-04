# Kernel_methods_codes
Transcription factors (TFs) have a crucial role in the control of genes. During the process of DNA
transcription, it transfers genetic information from messenger RNA to DNA. furthermore, the transcription
factor binds to a section of the DNA sequence known as Transcription Factor Binding Sites during this
stage.
In this project, our goal was to predict whether the sequence of DNA is binding site for a specific TF or
no by implementing an efficient machine learning algorithm. To achieve this, we have tried several machine
learning models(logistic regression, SVM) with different kernels (linear, gaussian, polynomial, spectrum
and mismatch). We found that mismatch Outperformed well with the accuracy of 66%.

# Data overview

The data used for this challenge consist of three datasets. For each of these datasets, we have 2000
labeled training sequences of 101 nucleotides (Xtr0.csv or Xtr0 mat100.csv, Xtr1.csv or Xtr1 mat100.csv,
Xtr2.csv or Xtr2 mat100.csv), as well as 1000 unlabeled test (Xte0.csv or Xte0 mat100.csv , Xte1.csv or
Xte1 mat100.csv, Xte2.csv, Xte2 mat100.csv) sequences that we want to classify. The Xtrk mat100.csv
contained numeral values. In a DNA sequence, We have four types of nucleotides: Adenine Base(A),
Thymine Base(T), Cytosine Base(C), Guanine Base(G).
In order to measure the quality of a model before submitting, we first performed training-validation
splits on the labeled datasets, with 80%-20% ratio (1600 samples for the training and 400 samples for the
validation).
