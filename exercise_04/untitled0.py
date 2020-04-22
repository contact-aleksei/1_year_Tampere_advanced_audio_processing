import csv, re
from collections import Counter

with open('clotho_captions_subset.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    columns_cap_1,columns_cap_2,columns_cap_3,columns_cap_4,columns_cap_5=[],[],[],[],[]
    threshold=14
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
                output_str ='<sos> '+output_str +' <eos>'+' <eos>'
            else:
                output_str ='<sos> '+' '.join(output_str.split()[:threshold])+' <eos>'
            columns_cap_1.append(output_str)
            all_words_string+=output_str+' '
            
            input_str=str(row[2])     
            output_str = re.sub('[^A-Za-z0-9]+', ' ', input_str)
            if len(output_str.split())==threshold:
                output_str ='<sos> '+output_str+' <eos>'
            elif len(output_str.split())<threshold:
                output_str ='<sos> '+output_str +' <eos>'+' <eos>'
            else:
                output_str ='<sos> '+' '.join(output_str.split()[:threshold])+' <eos>'
            columns_cap_2.append(output_str)
            all_words_string+=output_str+' '
            
            input_str=str(row[3])     
            output_str = re.sub('[^A-Za-z0-9]+', ' ', input_str)
            if len(output_str.split())==threshold:
                output_str ='<sos> '+output_str+' <eos>'
            elif len(output_str.split())<threshold:
                output_str ='<sos> '+output_str +' <eos>'+' <eos>'
            else:
                output_str ='<sos> '+' '.join(output_str.split()[:threshold])+' <eos>'
            columns_cap_3.append(output_str)
            all_words_string+=output_str+' '
            
            input_str=str(row[4])     
            output_str = re.sub('[^A-Za-z0-9]+', ' ', input_str)
            if len(output_str.split())==threshold:
                output_str ='<sos> '+output_str+' <eos>'
            elif len(output_str.split())<threshold:
                output_str ='<sos> '+output_str +' <eos>'+' <eos>'
            else:
                output_str ='<sos> '+' '.join(output_str.split()[:threshold])+' <eos>'
            columns_cap_4.append(output_str)
            all_words_string+=output_str+' '
            
            input_str=str(row[5])     
            output_str = re.sub('[^A-Za-z0-9]+', ' ', input_str)
            if len(output_str.split())==threshold:
                output_str ='<sos> '+output_str+' <eos>'
            elif len(output_str.split())<threshold:
                output_str ='<sos> '+output_str +' <eos>'+' <eos>'
            else:
                output_str ='<sos> '+' '.join(output_str.split()[:threshold])+' <eos>'
            columns_cap_5.append(output_str)
            all_words_string+=output_str+' '
           
            line_count += 1

all_words_string=all_words_string.lower()
unique_words = list(set(all_words_string.split(' ')))
unique_words.remove('<sos>')
unique_words.remove('<eos>')

 