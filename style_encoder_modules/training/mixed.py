import torch
from tqdm import tqdm

from .meters import AvgMeter
from .losses import performance


############################ MIXED TRAINING ############################################              
def train_epoch_mixed(train_loader, model, criterion_triplet, criterion_classification, optimizer, device, args):
    
    model.train()
    running_loss = 0
    total = 0
    n_corrects = 0
    loss_meter = AvgMeter()
    loss_meter_triplet = AvgMeter()
    loss_meter_class = AvgMeter()
    pbar = tqdm(train_loader)
    for i, data in enumerate(pbar):
        
        img = data[0]
        wid = data[3].to(device)
        positive = data[4].to(device)
        negative = data[5].to(device)
        
        anchor = img.to(device)
        # Get logits and features from the model
        anchor_logits, anchor_features = model(anchor)
        _, positive_features = model(positive)
        _, negative_features = model(negative)
        
        _, preds = torch.max(anchor_logits.data, 1)
        n_corrects += (preds == wid.data).sum().item()
    
        classification_loss = performance(anchor_logits, wid)
        triplet_loss = criterion_triplet(anchor_features, positive_features, negative_features)
        
        
        loss = classification_loss + triplet_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        #pbar.set_postfix(triplet_loss=loss.item())
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        loss_meter_triplet.update(triplet_loss.item(), count)
        loss_meter_class.update(classification_loss.item(), count)
        pbar.set_postfix(mixed_loss=loss_meter.avg, classification_loss=loss_meter_class.avg, triplet_loss=loss_meter_triplet.avg)
        total += img.size(0)
    
    accuracy = n_corrects/total
    print('total', total)
    print("Training Loss: {:.4f}".format(running_loss/len(train_loader)))
    print("Training Accuracy: {:.4f}".format(accuracy*100))
    return running_loss/total #np.mean(running_loss)/total


def val_epoch_mixed(val_loader, model, criterion_triplet, criterion_classification, optimizer, device, args):
    
    running_loss = 0
    total = 0
    n_corrects = 0
    loss_meter = AvgMeter()
    pbar = tqdm(val_loader)
    for i, data in enumerate(pbar):
        
        img = data[0].to(device)
        wid = data[3].to(device)
        positive = data[4].to(device)
        negative = data[5].to(device)
        
        anchor = img
        anchor_logits, anchor_features = model(anchor)
        _, positive_features = model(positive)
        _, negative_features = model(negative)
        
        _, preds = torch.max(anchor_logits.data, 1)
        n_corrects += (preds == wid.data).sum().item()
    
        classification_loss = performance(anchor_logits, wid)
        triplet_loss = criterion_triplet(anchor_features, positive_features, negative_features)
        
        loss = classification_loss + triplet_loss
        
        #running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        pbar.set_postfix(mixed_loss=loss_meter.avg)
        total += wid.size(0)
    
    print('total', total)
    accuracy = n_corrects/total
    print("Validation Loss: {:.4f}".format(running_loss/len(val_loader)))
    print("Validation Accuracy: {:.4f}".format(accuracy*100))
    return running_loss/total #np.mean(running_loss)/total


#TRAINING CALLS
def train_mixed(model, train_loader, val_loader, criterion_triplet, criterion_classification, optimizer, scheduler, device, args):
    best_loss = float('inf')
    for epoch_i in range(args.epochs):
        model.train()
        train_loss = train_epoch_mixed(train_loader, model, criterion_triplet, criterion_classification, optimizer, device, args)
        print("Epoch: {}/{}".format(epoch_i+1, args.epochs))
        
        model.eval()
        with torch.no_grad():
            val_loss = val_epoch_mixed(val_loader, model, criterion_triplet, criterion_classification, optimizer, device, args)
        
        if val_loss < best_loss:
            best_loss =val_loss
            torch.save(model.state_dict(), f'{args.save_path}/mixed_{args.dataset}_{args.model}.pth')
            print("Saved Best Model!")
        
        scheduler.step(val_loss)
