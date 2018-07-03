
import win_unicode_console
win_unicode_console.enable()

from PIL import Image
from keras.models import load_model
import os
import numpy as np
import operator

model = load_model('model.h5')
model.summary()

#path=r".\testData"
path=r".\Face Database"

total = 0
firsttrue = 0
false = 0

for filename in os.listdir(path):
    if(filename[0] == 's'):
        test = Image.open(path+'\\'+filename)
        Ans = int(filename[1]) * 10 + int(filename[2])
        matrix = np.array(test) / 255  
        matrix = matrix[np.newaxis,...]
        result = model.predict(matrix)
        
        result_dict = {}
        for i in range(len(result[0])):
            result_dict[i] = float(result[0][i])
        sorted_result_dict = sorted(result_dict.items(), key=operator.itemgetter(1), reverse=True)

        print("\nPredict Answer:")
        print(sorted_result_dict[0][0])
        print("\ncurrect Answer:")
        print(Ans)

        print('\nTop 5 candidate: ')
        count = 0
        ansnum = 0
        for topf in sorted_result_dict:
            count += 1
            if count > 5:
                break
            print(str(topf[0])+' Possibility: %0.8f' %(topf[1]))

            if(topf[0]==Ans): ansnum = count
        
        if(ansnum == 0):
            print("\nFalse")
            false+=1
        else:
            print("\nTrue %i" %(ansnum))

            if(ansnum == 1): firsttrue+=1

        total+=1
    
print("\nTotal Test data : %i"  %(total))    
print("\nFirst True data : %i"  %(firsttrue))
print("\n     False data : %i"  %(false))

print("\n                  %i%%" %((total-false)/total * 100 )   )