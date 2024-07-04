import tqdm
import torch,torch.nn as nn 
import numpy as np
from tqdm import tqdm

def train_loop(model,train_loader,test_loader,optimizer,loss_fn,lr_schedule=None,EPOCHS=10,device='cpu',run=None):

    best_loss = float('inf')
    for epoch in tqdm(range(1, EPOCHS + 1)):
                
        model.train()
        train_loss = 0
        train_acc = 0
        
        for X,y in train_loader:
            
            X,y = X.to(device),y.to(device)
            
            logits = model(X)
            loss = loss_fn(logits,y)
            
            train_loss+=loss
            
            preds = torch.argmax(logits,dim=-1)
            acc = (preds==y).sum().item()
            train_acc +=acc
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            
            
            
        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader.dataset)
        
        print(f'Train_loss: {train_loss.item():.3f},Train_acc: {train_acc*100:.2f}%')
        
        test_loss = 0
        test_acc = 0
        model.eval()

        for X,y in test_loader:
            
            X,y = X.to(device),y.to(device)
            
            with torch.inference_mode():
                logits = model(X)
                
            loss = loss_fn(logits,y)
            
            test_loss+=loss
            
            preds = torch.argmax(logits,dim=-1)
            acc = (preds==y).sum().item()
            test_acc +=acc
        
        test_loss = test_loss / len(test_loader)
        test_acc = test_acc / len(test_loader.dataset)
        
        print(f'test_loss: {test_loss.item():.3f},test_acc: {test_acc*100:.2f}%')    
        

        if run:

            run['train/accuracy'].append(train_acc*100)
            run['test/accuracy'].append(test_acc*100)
            run['train/loss'].append(train_loss)
            run['test/loss'].append(test_loss)

            if lr_schedule:

                run['metrics/learning_rate'].append(lr_schedule.get_last_lr())

        if test_loss < best_loss:
            best_loss = test_loss
            
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved best model with test loss: {best_loss}')

        if epoch == EPOCHS:
            
            run['model_checkpoints'].upload('best_model.pth')
            print('Uploaded best model to run')

        if lr_schedule:
            lr_schedule.step()
