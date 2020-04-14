import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from dataset import *
import torch.utils.data
from dataset.dataset_cifar import *
from dataset.data_preprocessing import  *
from models.models_vgg import *
import models.models_vgg

class imbalanced_VGG16() :
    def __init__(self, trainset=None, testset=None, lr=None, batch_norm=None, imbalance_ratio = None,  mu = None) :
        self.trainset= trainset
        self.testset = testset
        self.lr = lr
        #self.epoch = epoch
        self.imbalance_ratio = imbalance_ratio
        self.mu = mu
        self.optimzier = None
        self.scheduler = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_norm = batch_norm
        if batch_norm==True :
            self.net = vgg16_bn().cuda()
        else :
            self.net = vgg16().cuda()

        self.transform_train = transforms.Compose([transforms.ToTensor(),
                                                    transforms.ToPILImage(),
                                                    transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])
                                                    ])

        self.transform_test = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])
                                                  ])

        self.epoch_val_acc_list = []
        self.epoch_val_loss_list = []

    def one_epoch_train(self, trainloader=None, current_epoch=None):
        print('\nEpoch: %d' % current_epoch)
        print("Training...")
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            set_seed(2020)
            inputs = inputs.float().cuda()
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            # basic cross entropy
            ce_loss = criterion(outputs, targets)
            loss = ce_loss
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if current_epoch % 5 == 4:
                if batch_idx % 100 == 99:  # print every 2000 mini-batches
                    print('Training | [%d, %5d] loss: %.3f total accuracy : %.3f' %
                      (current_epoch + 1, batch_idx + 1, train_loss / (batch_idx + 1), correct / total))


    def one_epoch_test(self, testloader=None, current_epoch=None):
        global best_acc
        criterion = nn.CrossEntropyLoss()

        print('\nEpoch: %d' % current_epoch)
        print("Testing...")

        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                print(self.scheduler.get_lr())
                inputs = inputs.float().cuda()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)

                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)

                class_correct = predicted.eq(targets).sum().item()
                correct += predicted.eq(targets).sum().item()
                # print(total)
                # print(outputs.shape)
                print('Test | [%d, %5d] loss: %.3f total accuracy : %.3f' %
                      (current_epoch + 1, batch_idx + 1, test_loss / (batch_idx + 1), correct / total))

        self.epoch_test_loss = test_loss / (batch_idx + 1)
        self.epoch_test_correct = correct / total

        self.epoch_val_acc_list.append(self.epoch_test_correct)
        self.epoch_val_loss_list.append(self.epoch_test_loss)

        # Save checkpoint.
        acc = 100. * correct / total

        if acc > best_acc:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': current_epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if self.batch_norm == True :
                torch.save(state, './checkpoint/ckpt_imbalanced_vgg16_bn_{}_{}.pth'.format(self.imbalance_ratio, self.mu))
            else :
                torch.save(state, './checkpoint/ckpt_imbalanced_vgg16_bn_{}_{}.pth'.format(self.imbalance_ratio, self.mu))

            best_acc = acc

    def run(self, epochs):
        global best_acc
        best_acc=0
        set_seed(2020)
        train_dataset = CustomDataset(self.trainset[0], self.trainset[1], transform=self.transform_train)
        test_dataset = CustomDataset(self.testset[0], self.testset[1], transform=self.transform_test)

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9,
                                   weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=[60, 120, 160], gamma=0.2)

        for epoch in range(epochs):
            self.one_epoch_train(trainloader, epoch)
            self.one_epoch_test(testloader, epoch)
            self.scheduler.step()


class balanced_VGG16() :
    def __init__(self, trainset=None, testset=None, lr=None, batch_norm=None) :
        set_seed(2020)
        self.trainset= trainset
        self.testset = testset
        self.lr = lr
        #self.epoch = epoch
        self.optimzier = None
        self.scheduler = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_norm = batch_norm
        if self.batch_norm == True:
            self.net = vgg16_bn().cuda()
        else:
            self.net = vgg16().cuda()
        self.transform_train = transforms.Compose([transforms.ToTensor(),
                                                    transforms.ToPILImage(),
                                                    transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])
                                                    ])

        self.transform_test = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])
                                                  ])

        self.epoch_val_acc_list = []
        self.epoch_val_loss_list = []

    def one_epoch_train(self, trainloader=None, current_epoch=None):
        print('\nEpoch: %d' % current_epoch)
        print("Training...")
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            set_seed(2020)
            inputs = inputs.float().cuda()
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            # basic cross entropy
            ce_loss = criterion(outputs, targets)
            loss = ce_loss
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if current_epoch % 5 == 4:
                if batch_idx % 100 == 99:  # print every 2000 mini-batches
                    print('Training | [%d, %5d] loss: %.3f total accuracy : %.3f' %
                      (current_epoch + 1, batch_idx + 1, train_loss / (batch_idx + 1), correct / total))


    def one_epoch_test(self, testloader=None, current_epoch=None):
        global best_acc
        criterion = nn.CrossEntropyLoss()

        print('\nEpoch: %d' % current_epoch)
        print("Testing...")

        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                print(self.scheduler.get_lr())
                inputs = inputs.float().cuda()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)

                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)

                class_correct = predicted.eq(targets).sum().item()
                correct += predicted.eq(targets).sum().item()
                # print(total)
                # print(outputs.shape)
                print('Test | [%d, %5d] loss: %.3f total accuracy : %.3f' %
                      (current_epoch + 1, batch_idx + 1, test_loss / (batch_idx + 1), correct / total))

        self.epoch_test_loss = test_loss / (batch_idx + 1)
        self.epoch_test_correct = correct / total

        self.epoch_val_acc_list.append(self.epoch_test_correct)
        self.epoch_val_loss_list.append(self.epoch_test_loss)

        # Save checkpoint.
        acc = 100. * correct / total

        if acc > best_acc:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': current_epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if self.batch_norm == True:
                torch.save(state, './checkpoint/ckpt_balanced_vgg16_bn.pth')
            else:
                torch.save(state, './checkpoint/ckpt_balanced_vgg16.pth')
            best_acc = acc

    def run(self, epochs):
        global best_acc
        best_acc=0
        set_seed(2020)
        train_dataset = CustomDataset(self.trainset[0], self.trainset[1], transform=self.transform_train)
        test_dataset = CustomDataset(self.testset[0], self.testset[1], transform=self.transform_test)


        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9,
                                   weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=[55, 85], gamma=0.2)

        for epoch in range(epochs):
            self.one_epoch_train(trainloader, epoch)
            self.one_epoch_test(testloader, epoch)
            self.scheduler.step()

class imbalanced_VGG16_10() :
    def __init__(self, trainset=None, testset=None, lr=None, batch_norm=None, imbalance_ratio = None,  mu = None) :
        self.trainset= trainset
        self.testset = testset
        self.lr = lr
        #self.epoch = epoch
        set_seed(2020)
        self.imbalance_ratio = imbalance_ratio
        self.mu = mu
        self.optimzier = None
        self.scheduler = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_norm = batch_norm
        if batch_norm==True :
            self.net = models.models_vgg.vgg11_10_bn().cuda()
        else :
            self.net = vgg16().cuda()

        self.transform_train = transforms.Compose([transforms.ToTensor(),
                                                    transforms.ToPILImage(),
                                                    transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                    ])

        self.transform_test = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                  ])

        self.epoch_val_acc_list = []
        self.epoch_val_loss_list = []

    def one_epoch_train(self, trainloader=None, current_epoch=None):
        print('\nEpoch: %d' % current_epoch)
        print("Training...")
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            set_seed(2020)
            inputs = inputs.float().cuda()
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            # basic cross entropy
            ce_loss = criterion(outputs, targets)
            loss = ce_loss
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if current_epoch % 5 == 4:
                if batch_idx % 100 == 99:  # print every 2000 mini-batches
                    print('Training | [%d, %5d] loss: %.3f total accuracy : %.3f' %
                      (current_epoch + 1, batch_idx + 1, train_loss / (batch_idx + 1), correct / total))


    def one_epoch_test(self, testloader=None, current_epoch=None):
        global best_acc
        criterion = nn.CrossEntropyLoss()

        print('\nEpoch: %d' % current_epoch)
        print("Testing...")

        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                print(self.scheduler.get_lr())
                inputs = inputs.float().cuda()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)

                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)

                class_correct = predicted.eq(targets).sum().item()
                correct += predicted.eq(targets).sum().item()
                # print(total)
                # print(outputs.shape)
                print('Test | [%d, %5d] loss: %.3f total accuracy : %.3f' %
                      (current_epoch + 1, batch_idx + 1, test_loss / (batch_idx + 1), correct / total))

        self.epoch_test_loss = test_loss / (batch_idx + 1)
        self.epoch_test_correct = correct / total

        self.epoch_val_acc_list.append(self.epoch_test_correct)
        self.epoch_val_loss_list.append(self.epoch_test_loss)

        # Save checkpoint.
        acc = 100. * correct / total

        if acc > best_acc:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': current_epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if self.batch_norm == True :
                torch.save(state, './checkpoint/ckpt_cifar10_imbalanced_vgg11_bn_{}_{}.pth'.format(self.imbalance_ratio, self.mu))
            else :
                torch.save(state, './checkpoint/ckpt_cifar10_imbalanced_vgg11_bn_{}_{}.pth'.format(self.imbalance_ratio, self.mu))

            best_acc = acc

    def run(self, epochs):
        global best_acc
        best_acc=0
        set_seed(2020)
        train_dataset = CustomDataset(self.trainset[0], self.trainset[1], transform=self.transform_train)
        test_dataset = CustomDataset(self.testset[0], self.testset[1], transform=self.transform_test)

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9,
                                   weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=[55,85], gamma=0.2)

        for epoch in range(epochs):
            self.one_epoch_train(trainloader, epoch)
            self.one_epoch_test(testloader, epoch)
            self.scheduler.step()


class balanced_VGG16_10() :
    def __init__(self, trainset=None, testset=None, lr=None, batch_norm=None) :
        set_seed(2020)
        self.trainset= trainset
        self.testset = testset
        self.lr = lr
        #self.epoch = epoch
        self.optimzier = None
        self.scheduler = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_norm = batch_norm
        if self.batch_norm == True:
            self.net = models.models_vgg.vgg11_10_bn().cuda()
        else:
            self.net = vgg16().cuda()
        self.transform_train = transforms.Compose([transforms.ToTensor(),
                                                    transforms.ToPILImage(),
                                                    transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                    ])

        self.transform_test = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                  ])

        self.epoch_val_acc_list = []
        self.epoch_val_loss_list = []

    def one_epoch_train(self, trainloader=None, current_epoch=None):
        print('\nEpoch: %d' % current_epoch)
        print("Training...")
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            set_seed(2020)
            inputs = inputs.float().cuda()
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            # basic cross entropy
            ce_loss = criterion(outputs, targets)
            loss = ce_loss
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if current_epoch % 5 == 4:
                if batch_idx % 100 == 99:  # print every 2000 mini-batches
                    print('Training | [%d, %5d] loss: %.3f total accuracy : %.3f' %
                      (current_epoch + 1, batch_idx + 1, train_loss / (batch_idx + 1), correct / total))


    def one_epoch_test(self, testloader=None, current_epoch=None):
        global best_acc
        criterion = nn.CrossEntropyLoss()

        print('\nEpoch: %d' % current_epoch)
        print("Testing...")

        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                print(self.scheduler.get_lr())
                inputs = inputs.float().cuda()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)

                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)

                class_correct = predicted.eq(targets).sum().item()
                correct += predicted.eq(targets).sum().item()
                # print(total)
                # print(outputs.shape)
                print('Test | [%d, %5d] loss: %.3f total accuracy : %.3f' %
                      (current_epoch + 1, batch_idx + 1, test_loss / (batch_idx + 1), correct / total))

        self.epoch_test_loss = test_loss / (batch_idx + 1)
        self.epoch_test_correct = correct / total

        self.epoch_val_acc_list.append(self.epoch_test_correct)
        self.epoch_val_loss_list.append(self.epoch_test_loss)

        # Save checkpoint.
        acc = 100. * correct / total

        if acc > best_acc:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': current_epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if self.batch_norm == True:
                torch.save(state, './checkpoint/ckpt_cifar10_balanced_vgg11_bn.pth')
            else:
                torch.save(state, './checkpoint/ckpt_cifar10_balanced_vgg11.pth')
            best_acc = acc

    def run(self, epochs):
        global best_acc
        best_acc=0
        set_seed(2020)
        train_dataset = CustomDataset(self.trainset[0], self.trainset[1], transform=self.transform_train)
        test_dataset = CustomDataset(self.testset[0], self.testset[1], transform=self.transform_test)


        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9,
                                   weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=[55, 85], gamma=0.2)

        for epoch in range(epochs):
            self.one_epoch_train(trainloader, epoch)
            self.one_epoch_test(testloader, epoch)
            self.scheduler.step()