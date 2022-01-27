# MULTI-CLASS-CLASSIFICATION-using-BERT
Implementation of a multi-class classification model for labelling text from Linear Algebra by Gilbert Strang

### Summary of steps:
->Pick book to use as source of dataset.
->Read and store required text from all pages in the book
->Identify and define noise in the text using a RegEx and clean the text data
->Identify labels
->Prepare 2 datasets: 
--->each row as text from a whole page
--->each row as a line 
->Prepare a labels list for both,randomize the dataset using sampling method (pandas.sample(frac=1) ) and export as csv files using Pandas API
->Train 2 separate deep learning models using BERT Transfer Learning over the 2 datasets and save the models
->Test the 2 models against test sets from both page and sentence datasets.

### create_dataset.ipynb
->Read text from ![image](https://user-images.githubusercontent.com/80392139/151307604-7d7cf29c-1b29-4a0b-9386-2f683259f9ba.png)
