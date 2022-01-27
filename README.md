# MULTI-CLASS-CLASSIFICATION-using-BERT
Implementation of a multi-class classification model for labelling text from Linear Algebra by Gilbert Strang

### Summary of steps: 
-> Pick book to use as source of dataset.<br/><br/>
-> Read and store required text from all pages in the book<br/><br/> 
->Identify and define noise in the text using a RegEx and clean the text data<br/><br/> 
->Identify labels<br/><br/> 
->Prepare 2 datasets: <br/><br/> 
--->each row as text from a whole page<br/>
--->each row as a line <br/><br/> 
->Prepare a labels list for both,randomize the dataset using sampling method (pandas.sample(frac=1) ) and export as csv files using Pandas API<br/><br/> 
->Train 2 separate deep learning models using BERT Transfer Learning over the 2 datasets and save the models<br/><br/> 
->Test the 2 models against test sets from both page and sentence datasets.<br/><br/> 

### create_dataset.ipynb
->Read text from "Linear_algebra_and_its_applications_by_Strang_G._z-lib.org.pdf" using **PyMuPDF**(https://pymupdf.readthedocs.io/en/latest/)
![image](https://user-images.githubusercontent.com/80392139/151307854-fa9d9844-9842-4880-ac18-1a248049dcee.png)<br/><br/>
->Special symbols, images and diagrams won't get captured properly and also because the model is a text processing mode, I didn't capture the effect of diagrams and images.<br/>
--->Examples of what I defined as noisy data from the text :<br/>"x","singlestring","12.12","2","22333231","multiple words string to check if its not getting captured as noise by the RegEx","2am","Chapter 2", "5 2", "A =","b==","������","= 3·5−**2·61·5−2·4 = 3\n−3 = −1" <br/>
--->**Regex module** : https://docs.python.org/3/library/re.html  <br/><br/>
->Found RegEx patterns to find each of these cases separately and then combined all of them as a single RegEx using | : <br/> 
` p2= r'^\d*$|^\d+\s?\d+$|^.\w$|^\w?\s?=+|^[a-zA-Z](?!\w)|^�*$|^\d*\.\d*$' ` <br/><br/>
->**cleanText(text, pat)**: <br/> 
---> **text** : whole page as a string ; **pat** : RegEx to clean the text<br/> 
---> splits string into multiple strings using newline character(\n) as the separator => .split('\n')<br/> 
---> using findall() method from re library to return a list of all strings identified as noise and use a list object variable as a reference to it.<br/> 
---> If a string is in the list then return its index value and replace with '' <br/> 
---> If whole list just consists of '' or if number of space separated strings are <4 (also considered as noise by me), then skip these lines<br/> 
---> Append updated string to a list<br/> 
---> Repeat the whole process for all pages<br/> 
![image](https://user-images.githubusercontent.com/80392139/151325950-aa186f56-881a-402b-9940-1f085c04929e.png)<br/> <br/> 
-> I've considered each chapter as a separate label so that meant the model will predict 8 labels total.<br/>
---> Create a label list for of length= number of pages for page model<br/>
---> Create a label list for of length= number of lines for sentence model<br/><br/>
->Create a dictionary and export dataset as a csv file using to_csv() method from the Pandas library<br/><br/>
->At the end, I've just used sampling to split the dataset and organize the data using os and shutil methods<br/><br/>

### Training and testing the Models <br/>
-> import required libraries: torch,numpy,pandas,shutil<br/>
---> **torch**: to utilize the hugging face library<br/>
---> **numpy**: to deal with numpy types such as 'Inf' to use in the training loop<br/>
---> **pandas**: to read,sample and modify datasets if required<br/>
---> **transformers**: to utilize BERT API<br/><br/>
-> in sentence_models.ipynb, I've used unique() and tolist() methods to get a list of all the labels.<br/>
---> in pages_models.ipynb, I've just copied this list so as to train both the models with the same label encoding for the target labels.<br/><br/>
-> Create a class myDataset, the purpose of this class is to process the dataset and return text encodings: 'input_ids', 'token_type_ids', 'attention_mask'<br/>
-> Now this is the dataset that's going to be used by BERT<br/>
--->As I'm going to process my data in batches, I will need attention_mask as well as input ids<br/>
--->token type ids are usually required when dealing with sentence pairs, but I've included it as just standard procedure. It is NOT required.<br/><br/>
-> pass the myDataset objects for test and val as parameter for DataLoader constructor, along with parameters specifying the number of batches, shuffling of data, number of workers.<br/>![image](https://user-images.githubusercontent.com/80392139/151388218-ac5bb4dd-9875-4e91-901c-5447f398e774.png)<br/>
-> in class BERTModel, the structure of and steps involved in the neural network are specified:<br/>
--->**structure:** BERT layer->Dropout layer(to avoid overfitting)->Linear layer which for this model will have 768 weights(BERT-base) connected to 8 nodes(labels).<br/>
---> initialize an object variable for this class.<br/><br/>
->**Training loop**:<br/> 
--->set val_loss_min (keep record of minimum validation loss)<br/>
--->create a loop for a specific number of epochs in which I've defined another loop which iterates through all batches in the DataLoader object. For each such iteration:<br/>
------>pass input_ids, attention_mask, token_type_ids to the BERTModel object, return outputs, calculate loss, backpropogate, adjust weights using an optimizer.<br/>
--->create another loop within the epochs loop for validation DataLoader object and calculate loss. If the new validation loss is less than the current value of val_loss_min, then set reference of val_loss_min to this value and save the optimizer and model state dictionaries using torch.save()<br/>
--->Repeat for all epochs<br/><br/>
->**Testing**:<br/>
---> follow the steps to create a DataLoader object the same as train and validation set.<br/>
---> compare outputs with actual labels and calculate loss<br/><br/>
### Final test for the project: <br/>
->Load the models using torch.load and test sentence and page models against both test sets<br/><br/>

### Saving and Loading model and optimizer state dictionaries: <br/>
![image](https://user-images.githubusercontent.com/80392139/151401630-5091c519-4104-4fef-8101-bf755bded144.png)<br/>
->This next piece of code is used within the training loop to save the best performing model weight:<br/>
![image](https://user-images.githubusercontent.com/80392139/151401700-62c3d461-4695-4bf8-960c-e86856b1c18a.png)<br/>
->The next piece of code is used to load model state dict: <br/>
![image](https://user-images.githubusercontent.com/80392139/151402903-bb7f8df5-d774-4116-ba7d-352d00b089f1.png)
