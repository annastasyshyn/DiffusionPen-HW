import torch
from tqdm import tqdm
import time

from .losses import performance


# ===================== Training ==========================================
def train_class_epoch(model, training_data, optimizer, args):
    """Epoch operation in training phase"""

    model.train()
    total_loss = 0
    n_corrects = 0
    total = 0
    pbar = tqdm(training_data)
    for i, data in enumerate(pbar):

        image = data[0].to(args.device)
        label = data[3].to(args.device).long()

        optimizer.zero_grad()

        output = model(image)

        loss = performance(output, label)
        _, preds = torch.max(output.data, 1)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total += label.size(0)
        n_corrects += (preds == label).sum().item()
        pbar.set_postfix(Loss=loss.item())

    loss = total_loss / total
    accuracy = n_corrects / total

    return loss, accuracy


def eval_class_epoch(model, validation_data, args):
    """Epoch operation in evaluation phase"""

    model.eval()

    total_loss = 0
    total = 0
    n_corrects = 0
    prediction_list = []
    results = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(validation_data)):

            image = data[0].to(args.device)
            image_paths = data[8]
            label = data[3].to(args.device).long()

            output = model(image)

            loss = performance(output, label)  # performance
            _, preds = torch.max(output.data, 1)

            total_loss += loss.item()
            n_corrects += (preds == label.data).sum().item()
            total += label.size(0)
            # prediction_list.append(preds)
            # write into a file the img_path and the prediction
            # with open('predictions.txt', 'a') as f:
            #     for i, p in enumerate(preds):
            #         f.write(f'{image_paths[i]},{p}\n')

    loss = total_loss / total
    accuracy = n_corrects / total

    return loss, accuracy


def train_classification(
    model, training_data, validation_data, optimizer, scheduler, device, args
):  # scheduler # after optimizer
    """Start training"""

    valid_accus = []
    num_of_no_improvement = 0
    best_acc = 0

    for epoch_i in range(args.epochs):
        print("[Epoch", epoch_i, "]")

        start = time.time()
        # wandb.log({'lr': scheduler.get_last_lr()})
        # print('Epoch:', epoch_i,'LR:', scheduler.get_last_lr())

        train_loss, train_acc = train_class_epoch(model, training_data, optimizer, args)
        print(
            "Training: {loss: 8.5f} , accuracy: {accu:3.3f} %, "
            "elapse: {elapse:3.3f} min".format(
                loss=train_loss, accu=100 * train_acc, elapse=(time.time() - start) / 60
            )
        )

        start = time.time()
        model_state_dict = model.state_dict()
        checkpoint = {"model": model_state_dict, "settings": args, "epoch": epoch_i}

        if validation_data is not None:
            val_loss, val_acc = eval_class_epoch(model, validation_data, args)
            print(
                "Validation: {loss: 8.5f} , accuracy: {accu:3.3f} %, "
                "elapse: {elapse:3.3f} min".format(
                    loss=val_loss, accu=100 * val_acc, elapse=(time.time() - start) / 60
                )
            )

            if val_acc > best_acc:

                print("- [Info] The checkpoint file has been updated.")
                best_acc = val_acc
                torch.save(
                    model.state_dict(),
                    f"{args.save_path}/{args.dataset}_classification_{args.model}.pth",
                )
                num_of_no_improvement = 0
            else:
                num_of_no_improvement += 1

            if num_of_no_improvement >= 10:

                print("Early stopping criteria met, stopping...")
                break
        else:
            torch.save(
                model.state_dict(),
                f"{args.save_path}/{args.dataset}_classification_{args.model}.pth",
            )

        step_metric = val_loss if validation_data is not None else train_loss
        scheduler.step(step_metric)
