from os import path
import torch
import torch.utils.tensorboard as tb


def test_logging(train_logger, valid_logger):

    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """
    
    global_step=0
    epoch=0
    ac=[]
    # This is a strongly simplified training loop
    for epoch in range(10):
        torch.manual_seed(epoch)
        ac.clear()
        
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            ac.append(dummy_train_accuracy.mean())
            train_logger.add_scalar('loss',float(dummy_train_loss),global_step=global_step)
            global_step+=1
            
        train_logger.add_scalar('accuracy',sum(ac)/len(ac),global_step=global_step)
        torch.manual_seed(epoch)
        a=[]
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            a.append(dummy_validation_accuracy.mean())
        valid_logger.add_scalar('accuracy',sum(a)/len(a),global_step=global_step)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
