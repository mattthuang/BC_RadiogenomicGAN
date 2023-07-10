<<<<<<< HEAD
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to get the learning rate
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


# Function to compute the loss value per batch of data
def loss_batch(loss_func, output, target, opt=None):
    with torch.autograd.detect_anomaly():
        loss = loss_func(output, target)  # get loss
        pred = output.argmax(dim=1, keepdim=True) # Get Output Class
        pred = pred.squeeze(dim=1)
        metric_b = pred.eq(target.view_as(pred)).sum().item()  # get performance metric

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b


# Compute the loss value & performance metric for the entire dataset (epoch)
def loss_epoch(model, loss_function, dataset_dl, opt=None):
    run_loss = 0.0
    t_metric = 0.0
    len_data = len(dataset_dl.dataset)

    # internal loop over dataset
    for xb, yb in dataset_dl:
        # move batch to device
        xb = xb.to(device)
        yb = yb.to(device)
        yb = yb.unsqueeze(1).float()

        # print(yb, yb.shape, "this is yb")
        output = model(xb)  # get model output
        # output = output.view(1,1)
        print("output shape", output.shape)
        # l = loss_function(output, yb)
        # print(l, "yes l")
        loss_b, metric_b = loss_batch(loss_function, output, yb, opt)  # get loss per batch
        run_loss += loss_b  # update running loss

        if metric_b is not None:  # update running metric
            t_metric += metric_b

    loss = run_loss / float(len_data)  # average loss value
    metric = t_metric / float(len_data)  # average metric value

=======
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to get the learning rate
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


# Function to compute the loss value per batch of data
def loss_batch(loss_func, output, target, opt=None):
    with torch.autograd.detect_anomaly():
        loss = loss_func(output, target)  # get loss
        pred = output.argmax(dim=1, keepdim=True) # Get Output Class
        pred = pred.squeeze(dim=1)
        metric_b = pred.eq(target.view_as(pred)).sum().item()  # get performance metric

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b


# Compute the loss value & performance metric for the entire dataset (epoch)
def loss_epoch(model, loss_function, dataset_dl, opt=None):
    run_loss = 0.0
    t_metric = 0.0
    len_data = len(dataset_dl.dataset)

    # internal loop over dataset
    for xb, yb in dataset_dl:
        # move batch to device
        xb = xb.to(device)
        yb = yb.to(device)
        yb = yb.unsqueeze(1).float()

        # print(yb, yb.shape, "this is yb")
        output = model(xb)  # get model output
        # output = output.view(1,1)
        print("output shape", output.shape)
        # l = loss_function(output, yb)
        # print(l, "yes l")
        loss_b, metric_b = loss_batch(loss_function, output, yb, opt)  # get loss per batch
        run_loss += loss_b  # update running loss

        if metric_b is not None:  # update running metric
            t_metric += metric_b

    loss = run_loss / float(len_data)  # average loss value
    metric = t_metric / float(len_data)  # average metric value

>>>>>>> 959c822f3e294e28bffabe791f7b4ec2e6720746
    return loss, metric