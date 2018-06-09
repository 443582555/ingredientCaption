"""
        # evaluate on validation set
        if (epoch+1) % VAL_FREQ == 0 and epoch != 0:
            val_loss = validate(val_loader, model, decoder)

            # check patience
            if val_loss >= best_val:
                valtrack += 1
            else:
                valtrack = 0
        """       
        """    
            if valtrack >= opts.patience:
                # we switch modalities
                opts.freeVision = opts.freeRecipe; opts.freeRecipe = not(opts.freeVision)
                # change the learning rate accordingly
                adjust_learning_rate(optimizer, epoch, opts)
             valtrack = 0
            """
            """
            # save the best model
            is_best = val_loss < best_val
            best_val = min(val_loss, best_val)
            
            """
            """
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val': best_val,
                'optimizer': optimizer.state_dict(),
                'valtrack': valtrack,
                'freeVision': opts.freeVision,
                'curr_val': val_loss,
            }, is_best)
            """
            print '** Validation: %f (best) - %d (valtrack)' % (best_val, valtrack)

        
        # Compute Loss and Optimizer and peform Back propogation on it.
        """
        loss = criterion(output[0], output[1], target_var[0])
        # measure performance and record loss
        cos_losses.update(loss.data[0], input[0].size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        """
    
    """
    print('Epoch: {0}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'vision ({visionLR}) - recipe ({recipeLR})\t'.format(
                   epoch, loss=cos_losses, visionLR=optimizer.param_groups[1]['lr'],
                   recipeLR=optimizer.param_groups[0]['lr']))
    """
