# MULTI-LABEL-CLASSIFICATION-using-BERT
Implementation of a multi-label classification model for labelling text from `Linear Algebra and its applications by Gilbert Strang`
## Possible approaches:
-> Use bag-of-words and apply statistical and probabilistic analysis to predict the text’s label.
<br/>
-> Use Word-2-vec or GloVe embeddings and use LSTMs or ML models such as Naïve Bayes or Support Vector Machine
<br/><br/>
## Approach rationale:
-> Hugging face library offers an API for a pre-trained BERT model. We can essentially import all the pre-trained weights and biases from this model and fine tune it using our custom dataset. These have been fine-tuned on a huge amount of data. Running this process on a single-GPU google collab notebook would be unfeasible and this is a major reason to use Transfer Learning. 
<br/>
-> BERT makes use of positional encodings instead of frequentist approaches such as bag-of-words/Word2Vec/GloVe. In word2vec for example, stop-words aren’t considered as they would add no meaning. These words would occur in any sentence for any label. However, this approach doesn’t account for lexical or referential ambiguity. For example, “Reading is on the map “ and “Reading a map” mean and imply different things. BERT positional encodings are made using a stack of 12 encoders. All that is left to do is a downstream task for classification (specific to our label set). Which makes our tasks considerably less dependent on resources and allows for a task specific fine-tuning process.
<br/><br/>
## Summary of steps: 
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

## Experiments:
-> Trained and tested both models for different frequency rates in the Dropout layer and learning rates. <br/>
---> **Outcomes**:<br/>
------> For page model dropout layer with drop rate of 0.3 gave the best avg accuracy and avg loss over. Drop rate between [0.5,1] had a bad performance.   <br/>
------> For sentence model dropout layer with drop rate of 0.6 gave the best avg accuracy and avg loss over. Drop rate between [0.3,0.5] and [0.9,1] had a bad performance.   <br/>
-> Checked accuracy and model performance for different batch sizes. <br/>
---> **Outcomes**:<br/>
------> Intuition: to check if bigger number of sample affect the avg accuracy of the model. <br/>
------> Conclusion: Regardless of the batch size, the avg <br/>
->Tested both models against each both test datasets (page and sentence). <br/>
---> **Outcomes**:<br/>
**Accuracy / Loss**
|Model name| Page    | Sentence|
| :---:    |  :---:  |  :---:  |
|Page      | 83.3333 / 0.10060729 | 43.54386543 / 0.39329978|
|Sentence  | 83.3333 /0.10403795 | 78.56200528 /0.17218246 |


## Key learnings: 
-> For a long input sequences to the model, a lower drop rate generalizes the training data in a better way, because the number of units set to 0 would also be high. Implying a higher drop rate would cause too much data loss resulting in high bias and wouldn’t fulfil the purpose of generalization either. On the other hand, for shorter input sequences, the rate can be greater than 0.5, because the amount of data being compromised will be less too. Also, the amount of sentence inputs to be used are much higher than page inputs. <br/>
-> Training over sentence data yields overall more accurate results than the model that trains over page dataset. This could be because of less truncation and applying optimization steps on smaller chunks for the same labels, thus, training it better for each label. Sentence also offers more randomized data, which reduces the chances of **selection bias** as well as **accidental bias**. <br/>

## Files
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
-> I've considered each chapter as a separate label so that means that the model will predict 8 labels total.<br/>
---> Create a label list of length= number of pages for page model<br/>
---> Create a label list of length= number of lines for sentence model<br/><br/>
->Create a dictionary and export dataset as a csv file using to_csv() method from the Pandas library<br/><br/>
->At the end, I've just used sampling to split the dataset and organize the data using os and shutil methods<br/><br/>

### Training and testing the Models <br/>
-> import required libraries: torch,numpy,pandas,shutil<br/>
---> **torch**: to utilize PyTorch API<br/>
---> **numpy**: to deal with numpy types such as 'Inf' to use in the training loop<br/>
---> **pandas**: to read,sample and modify datasets if required<br/>
---> **transformers**: to utilize BERT API from the hugging face library<br/><br/>
-> in sentence_models.ipynb, I've used unique() and tolist() methods to get a list of all the labels.<br/>
---> in pages_models.ipynb, I've just copied this list so as to train both the models with the same label encoding for the target labels.<br/><br/>
-> Create a class myDataset, the purpose of this class is to process the dataset and return text encodings: 'input_ids', 'token_type_ids', 'attention_mask'<br/>
-> Now this is the dataset that's going to be used by BERT<br/>
--->As I'm going to process my data in batches, I will need attention_mask as well as input ids<br/>
--->token type ids are usually required when dealing with sentence pairs, but I've included it just as standard procedure. It is not really required.<br/><br/>
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
---> follow the steps to create a DataLoader object for test-set in the same way as train and validation set.<br/>
---> compare outputs with actual labels and calculate loss<br/><br/>
### Final test for the project: <br/>
->Load the models using torch.load and test sentence and page models against both test sets<br/><br/>

### Saving and Loading model and optimizer state dictionaries: <br/>
![image](https://user-images.githubusercontent.com/80392139/151401630-5091c519-4104-4fef-8101-bf755bded144.png)<br/>
->This next piece of code is used within the training loop to save the best performing model weight:<br/>
![image](https://user-images.githubusercontent.com/80392139/151401700-62c3d461-4695-4bf8-960c-e86856b1c18a.png)<br/>
->The next piece of code is used to load model state dict: <br/>
![image](https://user-images.githubusercontent.com/80392139/151402903-bb7f8df5-d774-4116-ba7d-352d00b089f1.png)
