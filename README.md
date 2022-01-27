# MULTI-CLASS-CLASSIFICATION-using-BERT
Implementation of a multi-class classification model for labelling text from Linear Algebra by Gilbert Strang

### Summary of steps:
->Pick book to use as source of dataset.<br/>
->Read and store required text from all pages in the book<br/>
->Identify and define noise in the text using a RegEx and clean the text data<br/>
->Identify labels<br/>
->Prepare 2 datasets: <br/>
--->each row as text from a whole page<br/>
--->each row as a line <br/>
->Prepare a labels list for both,randomize the dataset using sampling method (pandas.sample(frac=1) ) and export as csv files using Pandas API<br/>
->Train 2 separate deep learning models using BERT Transfer Learning over the 2 datasets and save the models<br/>
->Test the 2 models against test sets from both page and sentence datasets.<br/>

### create_dataset.ipynb
->Read text from "Linear_algebra_and_its_applications_by_Strang_G._z-lib.org.pdf" using PyMuPDF(https://pymupdf.readthedocs.io/en/latest/)
![image](https://user-images.githubusercontent.com/80392139/151307854-fa9d9844-9842-4880-ac18-1a248049dcee.png)
