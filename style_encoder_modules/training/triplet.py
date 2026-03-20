import torch
from tqdm import tqdm

from .meters import AvgMeter


########################################################################              
def train_epoch_triplet(train_loader, model, criterion, optimizer, device, args):
    
    model.train()
    running_loss = 0
    total = 0
    loss_meter = AvgMeter()
    pbar = tqdm(train_loader)
    for i, data in enumerate(pbar):
        
        img = data[0]
    
        if args.dataset == 'iam':
            wid = data[2]
            #print('wid', wid)
            positive = data[3]
            negative = data[4]
        
        anchor = img.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)
        
        loss = criterion(anchor_out, positive_out, negative_out)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        #pbar.set_postfix(triplet_loss=loss.item())
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        pbar.set_postfix(triplet_loss=loss_meter.avg)
        total += img.size(0)
    
    print('total', total)
    print("Training Loss: {:.4f}".format(running_loss/len(train_loader)))
    return running_loss/total #np.mean(running_loss)/total


def val_epoch_triplet(val_loader, model, criterion, optimizer, device, args):
    
    running_loss = 0
    total = 0
    pbar = tqdm(val_loader)
    for i, data in enumerate(pbar):
        
        img = data[0]
        #transcr = data[1]

        if args.dataset == 'iam':
            wid = data[2]
            positive = data[3]
            negative = data[4]
       
        anchor = img.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
    
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)
        
        loss = criterion(anchor_out, positive_out, negative_out)
        
        #running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        pbar.set_postfix(triplet_loss=loss.item())
        total += wid.size(0)
    
    print('total', total)
    print("Validation Loss: {:.4f}".format(running_loss/len(val_loader)))
    return running_loss/total #np.mean(running_loss)/total


def train_triplet(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args):
    best_loss = float('inf')
    for epoch_i in range(args.epochs):
        model.train()
        train_loss = train_epoch_triplet(train_loader, model, criterion, optimizer, device, args)
        print("Epoch: {}/{}".format(epoch_i+1, args.epochs))

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = val_epoch_triplet(val_loader, model, criterion, optimizer, device, args)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f'{args.save_path}/triplet_{args.dataset}_{args.model}.pth')

            scheduler.step()
        else:
            torch.save(model.state_dict(), f'{args.save_path}/triplet_{args.dataset}_{args.model}.pth')
            scheduler.step(train_loss)
