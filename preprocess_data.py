from cmath import pi
import pickle
import pandas as pd
import numpy as np
from dataset import DATA, PID_DATA
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable, grad
from model import KT_backbone
from dataset import DATA, PID_DATA
from sklearn.metrics import roc_auc_score
from utils import KTLoss, _l2_normalize_adv
from pytorchtools import EarlyStopping
from scipy.stats import pearsonr

device = torch.device("cpu")

with open('/home/thales/ATKT/dataset/errex/errex_dropped.csv','rb') as file:
   df = pd.read_csv(file)





with open('/home/thales/ATKT/student_interations.pickle','rb') as file:
   student_interations = pickle.load(file)

with open('/home/thales/ATKT/problems_per_skills.pickle','rb') as file:
   problems_per_skills = pickle.load(file)





#load model

skill_emb_dim = 256
answer_emb_dim = 96
hidden_emb_dim = 80
n_skill_dim = 4


net = KT_backbone(skill_emb_dim,answer_emb_dim,hidden_emb_dim,n_skill_dim)
net.load_state_dict(torch.load('/home/thales/ATKT/Checkpoints/atkt_pid.pt',map_location=torch.device('cpu')))



net.eval()

dat = PID_DATA(n_question=4,seqlen=250,separate_char=',',maxstep=500)

train_skill_data,train_answer_data = dat.load_data('/home/thales/ATKT/dataset/errex_pid/errex_pid_train1.csv')


print(train_skill_data.shape)

#print(train_skill_data[1])
#print(train_answer_data[1])

'''
with torch.no_grad():

   skill = torch.LongTensor(train_skill_data)
   answer = torch.LongTensor(train_answer_data)

   skill = torch.where(skill==-1, torch.tensor([3]), skill)
   answer = torch.where(answer==-1, torch.tensor([2]), answer)
   skill, answer = skill.to(device),answer.to(device)

   print(skill.shape)
   print(answer.shape)
   print(skill[0].shape)
   print(answer[0].shape)
   pred_res = net.predict(skill,answer)

   with open('preds.pickle','wb') as file:
      pickle.dump({'preds':pred_res},file)

'''

with open('/home/thales/ATKT/preds.pickle','rb') as file:
   preds = pickle.load(file)['preds']

preds = preds.cpu().detach().numpy()


students = df['student_id'].unique()
skills = df['skill_name']

student_mean_of_correctness = {}
for i in range(len(students)):

   student_seq =  train_skill_data[i]+1

   index_ordDecimals = np.where(student_seq==1)
   index_placeNumber = np.where(student_seq==2)
   index_completeSeque = np.where(student_seq ==3)
   index_decimaAddition = np.where(student_seq==4)

   mean_of_correctness = {}
   mean_of_correctness['OrderingDecimals'] = np.mean(preds[i][index_ordDecimals])
   mean_of_correctness['PlacementOnNumberLine'] = np.mean(preds[i][index_placeNumber])
   mean_of_correctness['CompleteTheSequence'] = np.mean(preds[i][index_completeSeque])
   mean_of_correctness['DecimalAddition'] = np.mean(preds[i][index_decimaAddition])

   student_mean_of_correctness[students[i]] = mean_of_correctness



#print(student_mean_of_correctness)




'''
students = df['student_id'].unique()
student_mean_of_correctness = {}

for student in students:
   problem = student_interations[student][0]
   problem_list = problem.tolist()
   answer = student_interations[student][1]
   with torch.no_grad():
      problem = torch.LongTensor(problem)
      answer = torch.LongTensor(answer)   
      problem = torch.unsqueeze(problem,dim=0)
      answer = torch.unsqueeze(answer,dim=0)
      pred_res = net.predict(problem,answer)

      skills_index = {}
      for problem in problems_per_skills.keys():
         index = []
         for i in problem_list:
            if i in problems_per_skills[problem]:
               index.append(problem_list.index(i))
         skills_index[problem] = index
      
      mean_of_correctness = {}
      for skill in skills_index.keys():
         mean_of_correctness[skill] = np.mean([pred_res[0][i] for i in skills_index[skill] ])

      student_mean_of_correctness[student] = mean_of_correctness



'''





ord_decimals = [i['OrderingDecimals'] for i in  list(student_mean_of_correctness.values())]
place_number = [i['PlacementOnNumberLine'] for i in  list(student_mean_of_correctness.values())]
complet_sequence = [i['CompleteTheSequence'] for i in  list(student_mean_of_correctness.values())]
decimal_addition =  [i['DecimalAddition'] for i in  list(student_mean_of_correctness.values())]

skills_score = {
    'OrderingDecimals':ord_decimals,
    'PlacementOnNumberLine':place_number,
    'CompleteTheSequence':complet_sequence,
    'DecimalAddition': decimal_addition
}


with open('/home/thales/ATKT/ErrEx posttest data.xlsx','rb') as file:
    df_2 = pd.read_excel(file)


df_2 = df_2.drop([0,1], axis=0)
df_2 = df_2.drop(df_2.columns[1:5],axis=1)


df_post_dropped = df_2.drop_duplicates('Anon Student Id',keep='last')
df_post_dropped = df_post_dropped.drop(df_post_dropped.index[598])

decimal_addition_post = df_post_dropped['Unnamed: 171'].values
ordering_decimals_post=  df_post_dropped['Unnamed: 172'].values 
complete_sequence_post = df_post_dropped['Unnamed: 173'].values 
placement_number_post = df_post_dropped['Unnamed: 174'].values 


decimal_addition_estimate = decimal_addition
placement_number_estimate = place_number
complete_sequence_estimate = complet_sequence
ordering_decimals_estimate = ord_decimals


pearson_correlations = {}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

pearson_correlations['OrderingDecimals'] = pearsonr(ordering_decimals_estimate,ordering_decimals_post)[0]
pearson_correlations['PlacementOnNumberLine'] = pearsonr(placement_number_estimate,placement_number_post)[0]
pearson_correlations['CompleteTheSequence'] = pearsonr(complete_sequence_estimate,complete_sequence_post)[0]
pearson_correlations['DecimalAddition'] = pearsonr(decimal_addition_estimate,decimal_addition_post)[0]

print(pearson_correlations)




   #TODO

   #tentar gerar para a última posição
   #carregar o problem per skill e tentar fazer de forma simular ao outro código 
   #calcular a média para cada aluno
   #computar a correlação com o post-test
