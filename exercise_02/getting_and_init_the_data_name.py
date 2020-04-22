import dataset_class
from answers import dataset_iteration

def mydatasets():
    training = dataset_class.MyDataset('training', 'music_speech_dataset')
    testing = dataset_class.MyDataset('testing', 'music_speech_dataset')
    validation =dataset_class.MyDataset('validation', 'music_speech_dataset')
    
    training=dataset_iteration(training, batch_size = 2,shuffle = True)
    testing=dataset_iteration(testing, batch_size = 2,shuffle = False)
    validation=dataset_iteration(validation, batch_size = 2,shuffle = True)

    return training, testing, validation
