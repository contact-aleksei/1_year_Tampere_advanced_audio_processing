import librosa
import numpy as np
import os, glob,ntpath
def reading_pickling(y,filename_to_save):
    c_x = librosa.stft(y, n_fft=2048, win_length=2048, hop_length=1024, window='hamm')
    seq_length = 60
    seq_length = len(c_x)
    chop_factor = c_x.shape[1] % seq_length
    new_time_frames = c_x.shape[1] - chop_factor    
    r_vectors = np.reshape(np.abs(c_x[:, :-chop_factor]), (new_time_frames//seq_length, c_x.shape[0], seq_length))    
    np.save(filename_to_save, r_vectors)
    return None


def give_path(path_original):
    path=path_original
    for filename in glob.glob(os.path.join(path, '*.wav')):
        y, sr = librosa.load(filename)  
        name=ntpath.basename(filename)[:-4]
        if 'testing' in filename:
            spl_word='testing\\'
            res = path_original.partition(spl_word)[2]
            number=len(res)
            path_original=path_original[:-number-len(spl_word)]+'testing_features'
            
            filename_to_save=path_original+'\\mix_'+name+'_seq_01'

            if os.path.isfile(filename_to_save+'.npy'):
                 filename_to_save=path_original+'\\mix_'+name+'_seq_02'
            else:
                filename_to_save=path_original+'\\mix_'+name+'_seq_01'
        else:
            spl_word='training\\'
            res = path_original.partition(spl_word)[2]
            number=len(res)
            path_original=path_original[:-number-len(spl_word)]+'training_features'
            filename_to_save=path_original+'\\mix_'+name+'_seq_01'
            
            if os.path.isfile(filename_to_save+'.npy'):
                 filename_to_save=path_original+'\\mix_'+name+'_seq_02'
            else:
                filename_to_save=path_original+'\\mix_'+name+'_seq_01'
            
        reading_pickling(y,filename_to_save)
    return None

give_path('C:\\Users\\OWNER\\Desktop\\studies TUT\\3 period\\Audio Processing\\exercise_05\\Mixtures\\testing\\testing_1')
give_path('C:\\Users\\OWNER\\Desktop\\studies TUT\\3 period\\Audio Processing\\exercise_05\\Mixtures\\testing\\testing_2')
give_path("C:\\Users\\OWNER\\Desktop\\studies TUT\\3 period\\Audio Processing\\exercise_05\\Mixtures\\training\\053 - Actions - Devil's Words")
give_path('C:\\Users\\OWNER\\Desktop\\studies TUT\\3 period\\Audio Processing\\exercise_05\\Mixtures\\training\\054 - Actions - South Of The Water')
give_path('C:\\Users\\OWNER\\Desktop\\studies TUT\\3 period\\Audio Processing\\exercise_05\\Sources\\testing\\testing_1')
give_path('C:\\Users\\OWNER\\Desktop\\studies TUT\\3 period\\Audio Processing\\exercise_05\\Sources\\testing\\testing_2')
give_path("C:\\Users\\OWNER\\Desktop\\studies TUT\\3 period\\Audio Processing\\exercise_05\\Sources\\training\\053 - Actions - Devil's Words")
give_path('C:\\Users\\OWNER\\Desktop\\studies TUT\\3 period\\Audio Processing\\exercise_05\\Sources\\training\\054 - Actions - South Of The Water')
    
    
    