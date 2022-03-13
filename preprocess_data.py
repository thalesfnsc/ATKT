import pandas as pd
import numpy as np
from dataset import DATA, PID_DATA


with open('/home/thales/ATKT/dataset/errex/errex_dropped.csv','rb') as file:
   df = pd.read_csv(file)


#dat = DATA(n_question=164,seqlen=250, separate_char=',', maxstep=250)
#train_skill_data, train_answer_data = dat.load_data('/home/thales/ATKT/dataset/errex/errex_train.csv')
#print(train_skill_data[0])

print(max(df['problem_id'].unique()))