import csv, re
from aux_functions import create_one_hot_encoding
import numpy as np
from torch.utils.data import Dataset, DataLoader



from itertools import chain


from torch import rand, Tensor, cat, zeros
from torch.cuda import is_available
from torch.nn import GRU, LSTM, Linear, MSELoss, Sigmoid, \
    ReLU, GRUCell, BCEWithLogitsLoss,CrossEntropyLoss
from torch.optim import Adam







class Dataset(Dataset):
    def __init__(self):
        super().__init__()
        with open("clotho_captions_subset.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0 
            columns_cap_1,columns_cap_2,columns_cap_3,columns_cap_4,columns_cap_5=[],[],[],[],[]
            threshold=13 # eventually it gives 15 words including <sos> <eos>
            all_words_string=''
            for row in csv_reader:
                
                if line_count == 0:
                    line_count += 1;
                else:          
                    input_str=str(row[1])
                    output_str =re.sub('[^A-Za-z0-9]+', ' ', input_str)
                    if len(output_str.split())==threshold:
                        output_str ='<sos> '+output_str+' <eos>'
                        
                    elif len(output_str.split())<threshold:
                        
                        output_str ='<sos> '+output_str
                        difference_to_replenish = threshold - len(output_str.split())+2             
                        for iterator in range(0,difference_to_replenish):
                            output_str=output_str + ' <eos>'
                            
                    else:
                        output_str ='<sos> '+' '.join(output_str.split()[:threshold])+' <eos>'
                        
                    columns_cap_1.append(output_str)
                    all_words_string+=output_str+' '
                    
                        
                    input_str=str(row[2])     
                    output_str = re.sub('[^A-Za-z0-9]+', ' ', input_str)
                    if len(output_str.split())==threshold:
                        output_str ='<sos> '+output_str+' <eos>'
                    elif len(output_str.split())<threshold:
                        
                        output_str ='<sos> '+output_str
                        difference_to_replenish = threshold - len(output_str.split())+2             
                        for iterator in range(0,difference_to_replenish):
                            output_str=output_str + ' <eos>'
                            
                    else:
                        output_str ='<sos> '+' '.join(output_str.split()[:threshold])+' <eos>'
                    columns_cap_2.append(output_str)
                    all_words_string+=output_str+' '  
                    input_str=str(row[3])     
                    output_str = re.sub('[^A-Za-z0-9]+', ' ', input_str)
                    if len(output_str.split())==threshold:
                        output_str ='<sos> '+output_str+' <eos>'
                    elif len(output_str.split())<threshold:
                        
                        output_str ='<sos> '+output_str
                        difference_to_replenish = threshold - len(output_str.split())+2             
                        for iterator in range(0,difference_to_replenish):
                            output_str=output_str + ' <eos>'
                            
                    else:
                        output_str ='<sos> '+' '.join(output_str.split()[:threshold])+' <eos>'
                    columns_cap_3.append(output_str)
                    all_words_string+=output_str+' '  
                    input_str=str(row[4])     
                    output_str = re.sub('[^A-Za-z0-9]+', ' ', input_str)
                    if len(output_str.split())==threshold:
                        output_str ='<sos> '+output_str+' <eos>'
                    elif len(output_str.split())<threshold:
                        
                        output_str ='<sos> '+output_str
                        difference_to_replenish = threshold - len(output_str.split())+2             
                        for iterator in range(0,difference_to_replenish):
                            output_str=output_str + ' <eos>'
                            
                    else:
                        output_str ='<sos> '+' '.join(output_str.split()[:threshold])+' <eos>'
                    columns_cap_4.append(output_str)
                    all_words_string+=output_str+' '
                    input_str=str(row[5])     
                    output_str = re.sub('[^A-Za-z0-9]+', ' ', input_str)
                    if len(output_str.split())==threshold:
                        output_str ='<sos> '+output_str+' <eos>'
                    elif len(output_str.split())<threshold:
                      
                        output_str ='<sos> '+output_str
                        difference_to_replenish = threshold - len(output_str.split())+2             
                        for iterator in range(0,difference_to_replenish):
                            output_str=output_str + ' <eos>'
                        
                    else:
                        output_str ='<sos> '+' '.join(output_str.split()[:threshold])+' <eos>'
                    columns_cap_5.append(output_str)              
        
        # Putting all the words into string as well           
                    all_words_string+=output_str+' '       
                    line_count += 1        
        # I though we need to put all 5 captions in 5 different lists, as it was done above
        # Here below_ we sum up all the caps into one list into "all_captions" variable
            self.all_captions=columns_cap_1+columns_cap_2+columns_cap_3+columns_cap_4+columns_cap_5
        # Unique words
            self.unique_words = list(set(all_words_string.split(' ')))

    def __len__(self):
        return len(self.all_captions)
    def __getitem__(self,sentence):          
        caption_input=[]
        caption_targeted_output=[]
        encoded_sentence=[]
        sentence = self.all_captions[sentence]
        for word in sentence.split():           
            hot_word=create_one_hot_encoding(word,self.unique_words)
            encoded_sentence.append(hot_word)       
        encoded_sentence=np.array( encoded_sentence)
        caption_input=np.delete(encoded_sentence, -1)
        caption_targeted_output=np.delete(encoded_sentence,0)       
        return  caption_input, caption_targeted_output
                
    
def myRNN():
    device = 'cuda' if is_available() else 'cpu'
    print(f'Using {device}')
    epochs = 200
    
    #1. Firstly, use a linear layer
    # number of of your unique words and output / embedding
    embedding= Linear(in_features=4945,out_features= 64).to(device)
    # 2. Create a three-layred GRU-based DNN, using input featur
    rnn = GRU(input_size=64, hidden_size=64, num_layers=3, batch_first=True).to(device)
    # 3. Then, create the classifier of
    classifier = Linear(in_features=64,out_features=4945).to(device)      
    # As a loss function, you will use the CrossEntropy loss
    loss_f = CrossEntropyLoss()   
    # 4. Use the Adam optimizer
    optimizer = Adam(chain(embedding.parameters(), rnn.parameters(),classifier.parameters()))  
   
    activation = Sigmoid()
    dataset = Dataset()
    training_generator = DataLoader(dataset, batch_size=2)

         
    for epoch in range(epochs):
        epoch_loss = []
        
# reading dataloader
        for local_batch, local_labels in training_generator:
         
            optimizer.zero_grad()
            x_in = local_batch
            y_out = local_labels
            y_hat = activation(classifier(rnn(embedding(x_in)[0])))

            loss = loss_f(y_hat, y_out)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()           
        print(f'Epoch: {epoch:03d} | Loss: {Tensor(epoch_loss).mean():7.4f}')        
k=myRNN()      
        
        
        
