import argparse
import time
import torch
from dataloader import get_loader, load_data, normlize_data
from model import GCNGenerator, Discriminator, CNNGenerator2, CNNGenerator1
from utils import *

from tensorboardX import SummaryWriter
import shutil
import pdb
from Loss_custom import Pearson_loss_regions, Pearson_loss_whole

# Device configuration
# device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")


def main(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    # Create model directory

    if os.path.exists(args.runs_path):
        shutil.rmtree(args.runs_path)
    os.makedirs(args.runs_path)

    if os.path.exists(args.model_path):
        shutil.rmtree(args.model_path)
    os.makedirs(args.model_path)

    if os.path.exists(args.results_path):
        shutil.rmtree(args.results_path)
    os.makedirs(args.results_path)

    if os.path.exists(args.middle_results_path):
        shutil.rmtree(args.middle_results_path)
    os.makedirs(args.middle_results_path)

    if args.atlas == 'atlas1':
    	empty_list = [41, 116]
    elif args.atlas == 'atlas2':
    	empty_list = [3, 38]

    all_data = load_data(args.data_path, empty_list)
    data_mean = normlize_data(all_data, empty_list)

    exp_prec = []
    for exp_num in range(2):
        # Build the models
        if os.path.exists(args.middle_results_path + '/' + str(exp_num)):
            shutil.rmtree(args.middle_results_path + '/' + str(exp_num))
        os.makedirs(args.middle_results_path + '/' + str(exp_num))

        generator = GCNGenerator(args.input_size, args.out1_feature, args.out2_feature, args.out3_feature, 0.6)
        discriminator = Discriminator(args.input_size, args.out1_feature, args.out2_feature, args.out3_feature, 0.6)

        test_generator = GCNGenerator(args.input_size, args.out1_feature, args.out2_feature, args.out3_feature, 0.6)
        test_discriminator = Discriminator(args.input_size, args.out1_feature, args.out2_feature, args.out3_feature,
                                           0.6)

        generator = generator.to(device)
        discriminator = discriminator.to(device)

        test_generator = test_generator.to(device)
        test_discriminator = test_discriminator.to(device)

        adversarial_loss = torch.nn.BCELoss()
        # adversarial_loss = torch.nn.MSELoss()

        optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2),
                                       weight_decay=args.weight_decay)
        scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'max', patience=args.patience)

        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate,
                                       betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, 'max', patience=args.patience)

        exp_log_dir = os.path.join(args.runs_path, str(exp_num))
        if not os.path.isdir(exp_log_dir):
            os.makedirs(exp_log_dir)
        writer = SummaryWriter(log_dir=exp_log_dir)

        best_prec = 1000000
        for epoch in range(args.num_epochs):
            train_data_loader = get_loader(args.data_path, all_data, data_mean, empty_list, True, False, args.batch_size,
                                           num_workers=args.num_workers)
            val_data_loader = get_loader(args.data_path, all_data, data_mean, empty_list, False, False, args.batch_size,
                                         num_workers=args.num_workers)

            if epoch < args.pre_epochs:
                pre_train(args, train_data_loader, generator, discriminator, adversarial_loss, optimizer_G,
                          optimizer_D, writer, epoch, exp_num, device)
            else:
                optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.learning_rate / 1,
                                               betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
                optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate / 1,
                                               betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

                train(args, train_data_loader, generator, discriminator, adversarial_loss, optimizer_G,
                      optimizer_D, writer, epoch, exp_num, device)
                prec = validate(args, val_data_loader, generator, discriminator, adversarial_loss, writer, epoch,
                                exp_num, device)
                scheduler_G.step(prec)
                scheduler_D.step(prec)

                # Save the model checkpoints
                if prec < best_prec:
                    best_prec = prec
                    torch.save(generator.state_dict(), os.path.join(
                        args.model_path, 'GAN_generator-{}.ckpt'.format(exp_num)))
                    torch.save(discriminator.state_dict(), os.path.join(
                        args.model_path, 'GAN_discriminator-{}.ckpt'.format(exp_num)))

        test_data_loader = get_loader(args.data_path, all_data, data_mean, empty_list, False, True, 1, num_workers=args.num_workers)
        test_prec = test(args, test_data_loader, test_generator, test_discriminator, adversarial_loss, writer, epoch, exp_num, device)
        print("Test Prec:", test_prec)
        exp_prec.append(test_prec)
        writer.close()

        del generator
        del discriminator
        del test_generator
        del test_discriminator
        del adversarial_loss
        del optimizer_G
        del optimizer_D

        del scheduler_G
        del scheduler_D
        del writer
    print(exp_prec)


def pre_train(args, data_loader, generator, discriminator, adversarial_loss, optimizer_G, optimizer_D, writer,
              epoch, exp_num, device):
    generator.train()

    if epoch % 50 == 0:
        for name, param in generator.named_parameters():
            if name == 'weight':
                print(param)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    start = time.time()
    for i, (subject, adj_matrix, func_matrix) in enumerate(data_loader):
        # pdb.set_trace()
        batchSize = adj_matrix.shape[0]
        data_time.update(time.time() - start)

        funcs = func_matrix.to(device).float()
        adjs_real = adj_matrix.to(device).float()

        adjs_gen = generator(funcs, funcs, batchSize, isTest=False)

        if epoch > 1:
            topo = adjs_gen
            adjs_gen = generator(topo, funcs, batchSize, isTest=False)

        loss = torch.nn.functional.mse_loss(adjs_gen, adjs_real) * args.nodes + Pearson_loss_regions(adjs_gen,
                                                                                              adjs_real) + Pearson_loss_whole(
            adjs_gen, adjs_real)
        loss.backward()
        optimizer_G.step()

        losses.update(loss.item())

        batch_time.update(time.time() - start)

        start = time.time()

        if epoch % 25 == 0:
            adjs_gen_middle = adjs_gen.squeeze().cpu().detach().numpy()
            for j in range(batchSize):
                np.savetxt(args.middle_results_path + '/' + str(exp_num) + '/' + subject[j] + '_adjs-gen_exp(' + str(
                    exp_num) + ')_epoch(' + str(epoch) + ').txt',
                           adjs_gen_middle[j], fmt='%.9f')

        if epoch == 0:
            adjs_real_middle = adjs_real.squeeze().cpu().detach().numpy()
            for j in range(batchSize):
                np.savetxt(args.middle_results_path + '/' + str(exp_num) + '/' + subject[j] + '_adjs-real_exp(' + str(
                    exp_num) + ').txt', adjs_real_middle[j], fmt='%.9f')

        if i % args.log_step == 0:
            print('Pre_Train Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(data_loader),
                                                                batch_time=batch_time,
                                                                data_time=data_time,
                                                                loss=losses, ))
    writer.add_histogram('GAN_pre_train_loss', losses.avg, epoch)


def train(args, data_loader, generator, discriminator, adversarial_loss, optimizer_G, optimizer_D, writer, epoch,
          exp_num, device):
    # Train the models
    # pdb.set_trace()
    generator.train()
    discriminator.train()

    if epoch % 50 == 0:
        for name, param in generator.named_parameters():
            if name == 'weight':
                print(param)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    G_losses = AverageMeter()  # loss (per word decoded)
    D_losses = AverageMeter()  # loss (per word decoded)
    gen_top5accs = AverageMeter()
    real_top5accs = AverageMeter()
    fake_top5accs = AverageMeter()

    start = time.time()
    for i, (subject, adj_matrix, func_matrix) in enumerate(data_loader):
        # pdb.set_trace()
        batchSize = adj_matrix.shape[0]
        data_time.update(time.time() - start)

        funcs = func_matrix.to(device).float()
        adjs_real = adj_matrix.to(device).float()

        valid = torch.ones((adjs_real.size(0),), dtype=torch.float).to(device)
        fake = torch.zeros((adjs_real.size(0),), dtype=torch.float).to(device)

        optimizer_D.zero_grad()
        discriminator.zero_grad()
        score_real = discriminator(adjs_real, batchSize)
        real_loss = adversarial_loss(score_real, valid)
        # real_loss = adversarial_loss(score_real, torch.randint(0,2,(adjs_real.size(0), 1)).to(device).float())
        real_acc = float((score_real.view(-1) > 0.5).sum().item()) / float(batchSize)

        adjs_gen = generator(funcs, funcs, batchSize, isTest=False)

        if epoch > 1:
            topo = adjs_gen
            adjs_gen = generator(topo, funcs, batchSize, isTest=False)

        score_fake = discriminator(adjs_gen.detach(), batchSize)
        fake_loss = adversarial_loss(score_fake, fake)
        # fake_loss = adversarial_loss(score_fake, torch.randint(0,2,(adjs_gen.size(0), 1)).to(device).float())
        d_loss = (real_loss + fake_loss) * 0.5

        fake_acc = float((score_fake.view(-1) < 0.5).sum().item()) / float(batchSize)

        if ((epoch - 200 + 5) % 5 == 0 and real_acc < 0.8) or ((epoch - 200 + 5) % 5 == 0 and fake_acc < 0.5):
            d_loss.backward()
            optimizer_D.step()
        # fake_acc = float((score_fake.view(-1) < 0.5).sum().item())/float(batchSize)

        optimizer_G.zero_grad()
        generator.zero_grad()
        g_score = discriminator(adjs_gen, batchSize)
        g_loss = adversarial_loss(g_score, valid) + torch.nn.functional.mse_loss(adjs_gen,adjs_real) * args.nodes + Pearson_loss_regions(adjs_gen, adjs_real) + Pearson_loss_whole(adjs_gen, adjs_real)
        # if itern%5 == 0:
        g_loss.backward()
        optimizer_G.step()
        gen_acc = float((g_score.view(-1) >= 0.5).sum().item()) / float(batchSize)

        # top5 = accuracy(torch.sigmoid(scores), labels, 1)
        # pdb.set_trace()

        G_losses.update(g_loss.item())
        D_losses.update(d_loss.item())

        real_top5accs.update(real_acc)
        fake_top5accs.update(fake_acc)
        gen_top5accs.update(gen_acc)

        batch_time.update(time.time() - start)

        start = time.time()

        if epoch % 25 == 0:
            adjs_gen_middle = adjs_gen.squeeze().cpu().detach().numpy()
            for j in range(batchSize):
                np.savetxt(args.middle_results_path + '/' + str(exp_num) + '/' + subject[j] + '_adjs-gen_exp(' + str(
                    exp_num) + ')_epoch(' + str(epoch) + ').txt',
                           adjs_gen_middle[j], fmt='%.9f')

        if epoch == 0:
            adjs_real_middle = adjs_real.squeeze().cpu().detach().numpy()
            for j in range(batchSize):
                np.savetxt(args.middle_results_path + '/' + str(exp_num) + '/' + subject[j] + '_adjs-real_exp(' + str(
                    exp_num) + ').txt', adjs_real_middle[j], fmt='%.9f')

        # Print log info
        if i % args.log_step == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'G_Loss {G_loss.val:.4f} ({G_loss.avg:.4f})\t'
                  'D_Loss {D_loss.val:.4f} ({D_loss.avg:.4f})\t'
                  'Gen Accuracy {gen_top5.val:.3f} ({gen_top5.avg:.3f})\t'
                  'Real Accuracy {real_top5.val:.3f} ({real_top5.avg:.3f})\t'
                  'Fake Accuracy {fake_top5.val:.3f} ({fake_top5.avg:.3f})'.format(epoch, i, len(data_loader),
                                                                                   batch_time=batch_time,
                                                                                   G_loss=G_losses,
                                                                                   D_loss=D_losses,
                                                                                   gen_top5=gen_top5accs,
                                                                                   real_top5=real_top5accs,
                                                                                   fake_top5=fake_top5accs, ))
    writer.add_histogram('GAN_train_G_loss', G_losses.avg, epoch)
    writer.add_histogram('GAN_train_D_loss', D_losses.avg, epoch)
    writer.add_histogram('GAN_train_gen_acc', gen_top5accs.avg, epoch)
    writer.add_histogram('GAN_train_real_acc', real_top5accs.avg, epoch)
    writer.add_histogram('GAN_train_fake_acc', fake_top5accs.avg, epoch)


def validate(args, data_loader, generator, discriminator, adversarial_loss, writer, epoch, exp_num, device):
    # Evaluate the models
    generator.train()
    discriminator.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    G_losses = AverageMeter()  # loss (per word decoded)
    D_losses = AverageMeter()  # loss (per word decoded)
    real_losses = AverageMeter()
    fake_losses = AverageMeter()
    gen_top5accs = AverageMeter()
    real_top5accs = AverageMeter()
    fake_top5accs = AverageMeter()
    D_top5accs = AverageMeter()

    start = time.time()
    for i, (subject, adj_matrix, func_matrix) in enumerate(data_loader):
        batchSize = adj_matrix.shape[0]

        data_time.update(time.time() - start)

        funcs = func_matrix.to(device).float()
        adjs_real = adj_matrix.to(device).float()

        valid = torch.ones((adjs_real.size(0),), dtype=torch.float).to(device)
        fake = torch.zeros((adjs_real.size(0),), dtype=torch.float).to(device)

        adjs_gen = generator(funcs, funcs, batchSize, isTest=False)

        if epoch > 1:
            topo = adjs_gen
            adjs_gen = generator(topo, funcs, batchSize, isTest=False)

        # Forward, backward and optimize
        with torch.no_grad():
            score_real = discriminator(adjs_real, batchSize)
            score_fake = discriminator(adjs_gen.detach(), batchSize)
            g_score = discriminator(adjs_gen, batchSize)
            g_loss = adversarial_loss(g_score, valid) + torch.nn.functional.mse_loss(adjs_gen,
                                                                                     adjs_real) * args.nodes + Pearson_loss_regions(
                adjs_gen, adjs_real) + Pearson_loss_whole(adjs_gen, adjs_real)
            real_loss = adversarial_loss(score_real, valid)
            fake_loss = adversarial_loss(score_fake, fake)
            d_loss = (real_loss + fake_loss) * 0.5
            # loss = criterion(torch.sigmoid(scores), onehot_labels)

        real_acc = float((score_real.view(-1) > 0.5).sum().item()) / float(batchSize)
        fake_acc = float((score_fake.view(-1) < 0.5).sum().item()) / float(batchSize)
        D_acc = (real_acc + fake_acc) / 2.0
        gen_acc = float((g_score.view(-1) >= 0.5).sum().item()) / float(batchSize)

        G_losses.update(g_loss.item())
        D_losses.update(d_loss.item())
        real_losses.update(real_loss.item())
        fake_losses.update(fake_loss.item())

        real_top5accs.update(real_acc)
        fake_top5accs.update(fake_acc)
        D_top5accs.update(D_acc)
        gen_top5accs.update(gen_acc)

        batch_time.update(time.time() - start)

        start = time.time()

        if epoch % 25 == 0:
            adjs_gen_middle = adjs_gen.squeeze().cpu().detach().numpy()
            for j in range(batchSize):
                np.savetxt(args.middle_results_path + '/' + str(exp_num) + '/' + subject[j] + '_adjs-gen_exp(' + str(
                    exp_num) + ')_epoch(' + str(epoch) + ').txt',
                           adjs_gen_middle[j], fmt='%.9f')

        if epoch == 300:
            adjs_real_middle = adjs_real.squeeze().cpu().detach().numpy()
            for j in range(batchSize):
                np.savetxt(args.middle_results_path + '/' + str(exp_num) + '/' + subject[j] + '_adjs-real_exp(' + str(
                    exp_num) + ').txt', adjs_real_middle[j], fmt='%.9f')

    print('Val Epoch: [{0}/{1}]\t'
          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'G_Loss {G_loss.val:.4f} ({G_loss.avg:.4f})\t'
          'D_Loss {D_loss.val:.4f} ({D_loss.avg:.4f})\t'
          'Gen Accuracy {gen_top5.val:.3f} ({gen_top5.avg:.3f})\t'
          'Real Accuracy {real_top5.val:.3f} ({real_top5.avg:.3f})\t'
          'Fake Accuracy {fake_top5.val:.3f} ({fake_top5.avg:.3f})'.format(len(data_loader), len(data_loader),
                                                                           batch_time=batch_time,
                                                                           G_loss=G_losses,
                                                                           D_loss=D_losses,
                                                                           gen_top5=gen_top5accs,
                                                                           real_top5=real_top5accs,
                                                                           fake_top5=fake_top5accs, ))
    writer.add_histogram('GAN_val_G_loss', G_losses.avg, epoch)
    writer.add_histogram('GAN_val_D_loss', D_losses.avg, epoch)
    writer.add_histogram('GAN_val_gen_acc', gen_top5accs.avg, epoch)
    writer.add_histogram('GAN_val_real_acc', real_top5accs.avg, epoch)
    writer.add_histogram('GAN_val_fake_acc', fake_top5accs.avg, epoch)
    return G_losses.avg


def test(args, data_loader, test_generator, test_discriminator, adversarial_loss, writer, epoch, exp_num, device):
    if os.path.exists(args.results_path + '/' + str(exp_num)):
        shutil.rmtree(args.results_path + '/' + str(exp_num))
    os.makedirs(args.results_path + '/' + str(exp_num))
    # Evaluate the models
    test_generator.load_state_dict(torch.load(os.path.join(args.model_path, 'GAN_generator-{}.ckpt'.format(exp_num))))
    test_generator.eval()

    test_discriminator.load_state_dict(
        torch.load(os.path.join(args.model_path, 'GAN_discriminator-{}.ckpt'.format(exp_num))))
    test_discriminator.eval()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    G_losses = AverageMeter()  
    D_losses = AverageMeter()  
    real_losses = AverageMeter()
    fake_losses = AverageMeter()
    D_top5accs = AverageMeter()
    gen_top5accs = AverageMeter()
    real_top5accs = AverageMeter()
    fake_top5accs = AverageMeter()

    start = time.time()
    for i, (subject, adj_matrix, func_matrix) in enumerate(data_loader):
        batchSize = adj_matrix.shape[0]

        data_time.update(time.time() - start)

        funcs = func_matrix.to(device).float()
        adjs_real = adj_matrix.to(device).float()

        valid = torch.ones((adjs_real.size(0),), dtype=torch.float).to(device)
        fake = torch.zeros((adjs_real.size(0),), dtype=torch.float).to(device)

        adjs_gen = test_generator(funcs, funcs, batchSize, isTest=True)

        if epoch > 1:
            topo = adjs_gen
            adjs_gen = test_generator(topo, funcs, batchSize, isTest=True)

        adjs_gen_final = adjs_gen.squeeze().cpu().detach().numpy()
        np.savetxt(args.results_path + '/' + str(exp_num) + '/' + subject[0] + '_adjs_gen_' + str(exp_num) + '.txt',
                   adjs_gen_final, fmt='%.9f')

        np.savetxt(args.results_path + '/' + str(exp_num) + '/' + subject[0] + '_adjs_real_' + str(exp_num) + '.txt',
                   adjs_real.squeeze().cpu().detach().numpy(), fmt='%.9f')

        # Forward, backward and optimiz
        with torch.no_grad():
            score_real = test_discriminator(adjs_real, batchSize)
            score_fake = test_discriminator(adjs_gen.detach(), batchSize)
            g_score = test_discriminator(adjs_gen, batchSize)
            # pdb.set_trace()
            g_loss = adversarial_loss(g_score.view(-1), valid) + torch.nn.functional.mse_loss(adjs_gen, adjs_real) * args.nodes + Pearson_loss_regions(adjs_gen, adjs_real) + Pearson_loss_whole(adjs_gen, adjs_real)
            real_loss = adversarial_loss(score_real.view(-1), valid)
            fake_loss = adversarial_loss(score_fake.view(-1), fake)
            d_loss = (real_loss + fake_loss) * 0.5
            # loss = criterion(torch.sigmoid(scores), onehot_labels)

        real_acc = float((score_real.view(-1) > 0.5).sum().item()) / float(batchSize)
        fake_acc = float((score_fake.view(-1) < 0.5).sum().item()) / float(batchSize)
        D_acc = (real_acc + fake_acc) / 2.0
        gen_acc = float((g_score.view(-1) >= 0.5).sum().item()) / float(batchSize)

        G_losses.update(g_loss.item())
        D_losses.update(d_loss.item())
        real_losses.update(real_loss.item())
        fake_losses.update(fake_loss.item())

        real_top5accs.update(real_acc)
        fake_top5accs.update(fake_acc)
        D_top5accs.update(D_acc)
        gen_top5accs.update(gen_acc)
        batch_time.update(time.time() - start)

        start = time.time()

    print('Test Epoch: [{0}/{1}]\t'
          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'G_Loss {G_loss.val:.4f} ({G_loss.avg:.4f})\t'
          'D_Loss {D_loss.val:.4f} ({D_loss.avg:.4f})\t'
          'Gen Accuracy {gen_top5.val:.3f} ({gen_top5.avg:.3f})\t'
          'Real Accuracy {real_top5.val:.3f} ({real_top5.avg:.3f})\t'
          'Fake Accuracy {fake_top5.val:.3f} ({fake_top5.avg:.3f})'.format(len(data_loader), len(data_loader),
                                                                           batch_time=batch_time,
                                                                           G_loss=G_losses,
                                                                           D_loss=D_losses,
                                                                           gen_top5=gen_top5accs,
                                                                           real_top5=real_top5accs,
                                                                           fake_top5=fake_top5accs, ))
    return D_top5accs.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-atlas', '--atlas', type=str, default='atlas1', help='path for data')

    parser.add_argument('--data_path', type=str, default='./data/HCP_1064_SC_FC_atlas1', help='path for data')

    parser.add_argument('--model_path', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--results_path', type=str, default='./results', help='path for generased adjs')
    parser.add_argument('--middle_results_path', type=str, default='./middle_results', help='path for generased middle adjs')
    parser.add_argument('--runs_path', type=str, default='./runs', help='path for runs')

    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1, help='step size for saving trained models')

    parser.add_argument('--nodes', type=int, default=68, help='number of regions, 148 for atlas1 and 68 for atlas2')
    parser.add_argument('--input_size', type=int, default=68, help='dimension of input feature, 148 for atlas1 and 68 for atlas2')
    parser.add_argument('--out1_feature', type=int, default=68, help='dimension of discriminator gcn1, 148 for atlas1 and 68 for atlas2')
    parser.add_argument('--out2_feature', type=int, default=256, help='dimension of discriminator gcn2, 256 for both atlas1 and atlas2')
    parser.add_argument('--out3_feature', type=int, default=68, help='dimension of discriminator gcn3, 148 for atlas1 and 68 for atlas2')

    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--pre_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=float, default=10)

    parser.add_argument('-gpu_id', '--gpu_id', type=int, default=0)

    args = parser.parse_args()

    if args.atlas == 'atlas1':
    	args.data_path = './data/HCP_1064_SC_FC_atlas1'
    	args.nodes = 148
    	args.input_size = 148
    	args.out1_feature = 148
    	args.out3_feature = 148
    	args.model_path = './atlas1/models'
    	args.results_path = './atlas1/results'
    	args.middle_results_path = './atlas1/middle_results'
    	args.runs_path = './atlas1/runs'
    elif args.atlas == 'atlas2':
    	args.data_path = './data/HCP_1064_SC_FC_atlas2'
    	args.nodes = 68
    	args.input_size = 68
    	args.out1_feature = 68
    	args.out3_feature = 68
    	args.model_path = './atlas2/models'
    	args.results_path = './atlas2/results'
    	args.middle_results_path = './atlas2/middle_results'
    	args.runs_path = './atlas2/runs'
    else:
    	print('wrong atlas type!')
    	exit()

    print(args)
    main(args)
