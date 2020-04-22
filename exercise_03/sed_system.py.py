from getting_and_init_the_data_answer import get_data_loader
from getting_and_init_the_data_answer import get_dataset
from my_cnn_system_answer import *
data_path ='sed_dataset'
batch_size = 4

training = get_data_loader(dataset=get_dataset('training', data_path),batch_size=batch_size, shuffle=True)
validation = get_data_loader(dataset=get_dataset('validation', data_path), batch_size=batch_size,shuffle=True)
testing = get_data_loader(dataset=get_dataset('testing', data_path),batch_size=batch_size,shuffle=False)
