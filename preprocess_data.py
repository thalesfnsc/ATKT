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
n_skill_dim = 237

net = KT_backbone(skill_emb_dim,answer_emb_dim,hidden_emb_dim,n_skill_dim)
net.load_state_dict(torch.load('/home/thales/ATKT/kt_model_best_0_93_AUC.pt'))
net.eval()


skill = student_interations['Stu_024e2d136ae62f0f7ed61221f784c3ad'][0]
answer = student_interations['Stu_024e2d136ae62f0f7ed61221f784c3ad'][1]


print(skill)

#skill = np.insert(skill,skill.shape,2)
#answer = np.insert(answer,answer.shape,2)


with torch.no_grad():
   skill = torch.LongTensor(skill)
   answer = torch.LongTensor(answer)
   
   skill = torch.unsqueeze(skill,dim=0)
   answer = torch.unsqueeze(answer,dim=0)
   pred_res,features = net(skill,answer) 
   print(pred_res[0])
   print(answer)

   #TODO

   #tentar gerar para a última posição
   #carregar o problem per skill e tentar fazer de forma simular ao outro código 
   #calcular a média para cada aluno
   #computar a correlação com o post-test
