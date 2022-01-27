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
![image](https://user-images.githubusercontent.com/80392139/151307854-fa9d9844-9842-4880-ac18-1a248049dcee.png)<br/>
->Special symbols, images and diagrams won't get captured properly and also because the model is a text processing mode, I didn't capture the effect of diagrams and images.<br/>
--->Examples of what I defined as noisy data from the text :<br/>"x","singlestring","12.12","2","22333231","multiple words string to check if its not getting captured as noise by the RegEx","2am","Chapter 2", "5 2", "A =","b==","������","= 3·5−**2·61·5−2·4 = 3\n−3 = −1" <br/>
--->Regex module : [https://docs.python.org/3/library/re.html] #breh <b/r>
->Found RegEx patterns to find each of these cases separately and then combined all of them as a single RegEx using | : <br/> 
` p2= r'^\d*$|^\d+\s?\d+$|^.\w$|^\w?\s?=+|^[a-zA-Z](?!\w)|^�*$|^\d*\.\d*$' ` <br/>
