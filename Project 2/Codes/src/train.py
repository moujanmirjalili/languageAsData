import torch
import math
from src.helper import write_list_to_file
import os 




def train(model,num_epochs,optimizer,criterion,data_loader,path_to_save_folder,train_run_label,vocab_size,device,evaluate_every,dev_dataloader,print_every=5):
    path_to_save_folder = os.path.join(path_to_save_folder,train_run_label)
    if not os.path.exists(path_to_save_folder):
        os.makedirs(path_to_save_folder)
    
    train_losses = []
    perplexities = []
    all_perplex = []
    
    
    all_losses = []
    all_perp = []
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}")
            all_losses.append(loss.item())
            if batch_idx % 50 == 0:
                torch.save(model.state_dict(), path_to_save_folder+"/temp_save")
            if batch_idx % evaluate_every == 0:
                perplexity_simple = evaluate(model, dev_dataloader,criterion,device,vocab_size)
                all_perplex.append(perplexity_simple)
                print("Validation perplexity: "+str(perplexity_simple))
                model.train()

                
                
        avg_loss = total_loss / len(data_loader)
        perplexity = math.exp(avg_loss)
        train_losses.append(avg_loss)
        perplexities.append(perplexity)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
    write_list_to_file("losses",train_losses,path_to_save_folder)
    write_list_to_file("step_losses",all_losses,path_to_save_folder)
    write_list_to_file("perplexities",perplexities,path_to_save_folder)
    write_list_to_file("all_perplex",all_perplex,path_to_save_folder)

    torch.save(model.state_dict(), path_to_save_folder+"/"+"model")
    torch.save(model,  path_to_save_folder+"/"+"model_full")
    return (all_losses,train_losses,perplexities,all_perplex)




def train_attention(model,num_epochs,optimizer,criterion,data_loader,path_to_save_folder,train_run_label,vocab_size,device,evaluate_every,dev_dataloader,print_every=5):
    print("Started Training")
    
    train_losses = []
    perplexities = []

    path_to_save_folder = os.path.join(path_to_save_folder,train_run_label)
    if not os.path.exists(path_to_save_folder):
        os.makedirs(path_to_save_folder)
    
    all_losses = []
    all_perp = []
    all_perplex = []
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        
        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x)
            
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}")
            all_losses.append(loss.item())
            if batch_idx % 50 == 0:
                torch.save(model.state_dict(), path_to_save_folder+"/temp_save")
            if batch_idx % evaluate_every == 0:
                perplexity_simple = evaluate_attention(model, dev_dataloader,criterion,device,vocab_size)
                all_perplex.append(perplexity_simple)
                print("Validation perplexity: "+str(perplexity_simple))
                model.train()

                 
                
        avg_loss = total_loss / len(data_loader)
        perplexity = math.exp(avg_loss)
        train_losses.append(avg_loss)
        perplexities.append(perplexity)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
    write_list_to_file("losses",train_losses,path_to_save_folder)
    write_list_to_file("step_losses",all_losses,path_to_save_folder)
    write_list_to_file("perplexities",perplexities,path_to_save_folder)
    write_list_to_file("all_perplex",all_perplex,path_to_save_folder)
    torch.save(model.state_dict(), path_to_save_folder+"/"+"model")
    torch.save(model,  path_to_save_folder+"/"+"model_full")
    return (all_losses,train_losses,perplexities,all_perplex)



def evaluate(model, dataloader,criterion,device,vocab_size): 
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    perplexity_simple = math.exp(avg_loss)
    return perplexity_simple


def evaluate_attention(model, dataloader,criterion,device,vocab_size): 
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)        
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    perplexity_simple = math.exp(avg_loss)
    return perplexity_simple
 