# encoding: utf-8

from utils.util_logger import get_current_logger
from utils.util_config import get_output_dir
from utils.util_file import mkdir_if_not_exist
from utils.util_config import empty_config_node
from utils.util_img import *
from models import save_model
from .plain_tester import do_test
import torch

def do_train(cfg, model, train_loader, valid_loader, optimizer, scheduler, loss_fn):
    device = 'cpu' if empty_config_node(cfg.TASK.DEVICES) else 'gpu'
    model_path = mkdir_if_not_exist([get_output_dir(cfg), 'model'])
    logger = get_current_logger(cfg)
    num_epochs = cfg.SOLVER.NUM_EPOCHS
    log_period = cfg.RECORD.LOG_PERIOD
    test_period = cfg.RECORD.TEST_PERIOD
    save_period = cfg.RECORD.SAVE_PERIOD
    trainer_step = 0

    logger.info('Begin of training.')
    for epoch in range(num_epochs):
        for (y, x, _, _) in train_loader:
            trainer_step += 1
            # (y, x): y = T(x) + noise
            y, x = y.to(device), x.to(device)

            optimizer.zero_grad()
            e = model(y)
            l = loss_fn(e, x)
            l.backward()
            optimizer.step()

            # todo: merge bn

            # training information
            if trainer_step % log_period == 0:
                if scheduler:
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                message = 'Train <epoch: {}/{}, iter: {}, lr: {:.3e}, loss: {:.3e}>.'.format(
                    epoch, num_epochs, trainer_step, current_lr, l.item())
                logger.info(message)

            # test model
            if trainer_step % test_period == 0:
                result = do_test(cfg, model, valid_loader)
                logger.info('Test <epoch: {}/{}, iter: {}> ' \
                            'result, average PSNR: {:<4.2f}dB, average SSIM: {:<4.2f}.'.format(
                            epoch, num_epochs, trainer_step, result['avg_psnr'], result['avg_ssim']))
            
            # save model
            if trainer_step % save_period == 0:
                save_model(model, os.path.join(model_path, str(trainer_step) + '.pt'))
                logger.info('Saving the model in step {}.'.format(trainer_step))

        # update leanring rate
        if scheduler:
            scheduler.step()

    result = do_test(cfg, model, valid_loader)
    logger.info('Test <final epoch, iter: {}> ' \
                'result, average PSNR: {:<4.2f}dB, average SSIM: {:<4.2f}.'.format(
                trainer_step, result['avg_psnr'], result['avg_ssim']))


    save_model(model, os.path.join(model_path, 'lastest.pt'))
    logger.info('End of training.')
