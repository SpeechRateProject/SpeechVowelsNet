import torch.optim as optim
import torch
import argparse
import Datasets
import numpy as np
import os
from model_speech_yolo import load_model, create_speech_model
from train_speech_yolo import TrainSpeechYolo
import yolo_vowels_loss


def build_model_name(args):
    """
    build the model name by the args given
    :param args: args from the model included optimizer, learning rate, c b k etc.
    :return: The full name created from the models arguments
    """
    args_dict = ["opt", "lr", "batch_size", "arc", "c_b_k", "noobject_conf", "obj_conf",
                 "coordinate", "class_conf", "loss_type"]
    full_name = ""
    for arg in args_dict:
        if arg == "c_b_k":
            config_params = args.c_b_k.split('_')
            full_name += "c_{}_b_{}_k_{}".format(config_params[0], config_params[1], config_params[2])
            continue
        full_name += str(arg) + "_" + str(getattr(args, arg)) + "_"

    return full_name + ".pth"


class SpeechYoloVowels:

    def __init__(self, data_folder_train,
                 data_folder_valid, num_workers=0):
        """
        The init function get the train and validation folders
        load the model (if exist, create new one if not) and build it
        :param data_folder_train: folder with the files for training , each file have .wav , .wrd
        :param data_folder_valid: folder with the files for validation , each file have .wav, .wrd
        :param num_workers: The number of workers that will be used to train the model and get the data,
                            you can change it by your machine capabilities
        """
        self.args = self.init_args(data_folder_train, data_folder_valid)
        config_params = self.args.c_b_k.split('_')
        param_names = ['C', 'B', 'K']  # C - cells, B - number of boxes per cell, K- number of keywords
        self.config_dict = {i: int(j) for i, j in zip(param_names, config_params)}

        # BUILD MODEL#
        self.speech_net, self.epoch, self.best_valid_loss, self.best_correct_ratio, \
        self.best_no_object_object_wrong_ratio = self.build_model()

        self.optimizer = self.init_optimizer()

        self.train_loader, self.val_loader = self.init_data_loaders(num_workers)

        self.loss = yolo_vowels_loss.YoloVowelsLoss(noobject_conf=self.args.noobject_conf, obj_conf=self.args.obj_conf,
                                                    coordinate=self.args.coordinate,
                                                    class_conf=self.args.class_conf, loss_type=self.args.loss_type)

    def init_args(self, data_folder_train,
                  data_folder_valid):
        """
        Initialize the arguments for the model and parse them
        The arguments can be changed in the 'default'
        for example: the c_b_k can change the c=num of cells, b=num of boxes and k=num of classes
        trained_model - for load old model and keep train it , etc.

        :param data_folder_train: folder with the files for training
        :param data_folder_valid: folder with the files for validation
        :return: return the arguments after parse
        """
        parser = argparse.ArgumentParser(description='train yolo model')
        parser.add_argument('--train_data', type=str, default=data_folder_train,
                            help='location of the train data')
        parser.add_argument('--val_data', type=str, default=data_folder_valid,
                            help='location of the validation data')
        parser.add_argument('--arc', type=str, default='VGG19',
                            help='arch method (LeNet, VGG11, VGG13, VGG16, VGG19)')
        parser.add_argument('--opt', type=str, default='adam',
                            help='optimization method: adam || sgd')
        parser.add_argument('--momentum', type=float, default='0.9',
                            help='momentum')
        # C - cells, B - number of boxes per cell, K- number of keywords
        parser.add_argument('--c_b_k', type=str, default='32_2_15', help='C B K parameters')

        parser.add_argument('--prev_classification_model', type=str,
                            default='pretraining_model/pre_trained_8.pth',
                            help='the location of the prev classification model')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='initial learning rate')
        parser.add_argument('--epochs', type=int, default=120,
                            help='upper epoch limit')
        parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                            help='batch size')
        parser.add_argument('--dropout', type=float, default=0.0,
                            help='dropout probability value')
        parser.add_argument('--seed', type=int, default=1245,
                            help='random seed')
        parser.add_argument('--cuda', action='store_true',
                            help='use CUDA')
        parser.add_argument('--patience', type=int, default=25, metavar='N',
                            help='how many epochs of no loss improvement should we wait before stop training')
        parser.add_argument('--log-interval', type=int, default=120, metavar='N',
                            help='report interval')
        parser.add_argument('--save_folder', type=str, default='trained_model',
                            help='path to save the final model')
        parser.add_argument('--save_file', type=str, default='',
                            help='filename to save the final model')
        parser.add_argument('--trained_yolo_model', type=str,
                            default='trained_model/opt_adam_lr_0.001_batch_size_32_arc_VGG19_c_32_b_2_k_15noobject_conf_0.5_obj_conf_1_coordinate_10_class_conf_1_loss_type_mse_.pth',
                            help='load model already trained by this script')
        parser.add_argument('--augment_data', action='store_true', help='add data augmentation')
        parser.add_argument('--noobject_conf', type=float, default=0.5,
                            help='noobject conf')
        parser.add_argument('--obj_conf', type=float, default=1,
                            help='obj conf')
        parser.add_argument('--coordinate', type=float, default=10,
                            help='coordinate')
        parser.add_argument('--class_conf', type=float, default=1,
                            help='class_conf')
        parser.add_argument('--loss_type', type=str, default="mse",
                            help='loss with abs or with mse (abs, mse)')
        parser.add_argument('--decision_threshold', type=float, default=0.25,
                            help=' object exist threshold')
        parser.add_argument('--iou_threshold', type=float, default=0.5,
                            help='high iou threshold')

        args = parser.parse_args()
        print(args)
        args.cuda = torch.cuda.is_available()
        torch.manual_seed(args.seed)

        return args

    def build_model(self):
        """
        The function create the model from the arguments or from the exist model given
        :return: The speech net model , and the values from the model loaded: epoch, best loss etc.
                (if not exist - initialize to infinity/zero as needed)
        """
        # init parameters
        best_valid_loss = np.inf
        best_correct_ratio = 0
        best_no_object_object_wrong_ratio = np.inf
        epoch = 1

        # build model
        if os.path.isfile(self.args.trained_yolo_model):  # model exists
            print("model found")
            speech_net, check_acc, check_epoch, correct_ratio, no_object_object_wrong_ratio = load_model(
                self.args.trained_yolo_model)
            best_valid_loss = check_acc
            epoch = check_epoch
            best_correct_ratio = correct_ratio
            best_no_object_object_wrong_ratio = no_object_object_wrong_ratio
            if self.args.cuda:
                best_valid_loss = best_valid_loss.cuda()

        else:
            speech_net = create_speech_model(self.args.prev_classification_model, self.args.arc,
                                             self.config_dict, self.args.dropout)

        if self.args.cuda:
            print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
            speech_net = torch.nn.DataParallel(speech_net).cuda()

        return speech_net, epoch, best_valid_loss, best_correct_ratio, best_no_object_object_wrong_ratio

    def init_data_loaders(self, num_workers=0):
        """
        Load the train and validation data sets by the files provided in the initialize
        :param num_workers: The number of workers that will be used to train the model and get the data,
                            you can change it by your machine capabilities
        :return: The train and validation datasets loaders
        """
        train_dataset = Datasets.SpeechYoloDataSet(classes_root_dir=self.args.train_data,
                                                   this_root_dir=self.args.train_data,
                                                   yolo_config=self.config_dict, augment=self.args.augment_data)
        val_dataset = Datasets.SpeechYoloDataSet(classes_root_dir=self.args.train_data,
                                                 this_root_dir=self.args.val_data,
                                                 yolo_config=self.config_dict)

        # sampler_train = Datasets.ImbalancedDatasetSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=self.args.cuda, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.args.batch_size, shuffle=None,
            num_workers=num_workers, pin_memory=self.args.cuda, sampler=None)

        return train_loader, val_loader

    def init_optimizer(self):
        """
        Initialize the optimizer by the arguments given to the model
        :return: optimizer
        """
        # define optimizer
        if self.args.opt.lower() == 'adam':
            optimizer = optim.Adam(self.speech_net.parameters(), lr=self.args.lr)
        elif self.args.opt.lower() == 'sgd':
            optimizer = optim.SGD(self.speech_net.parameters(), lr=self.args.lr,
                                  momentum=self.args.momentum)
        else:
            optimizer = optim.SGD(self.speech_net.parameters(), lr=self.args.lr,
                                  momentum=self.args.momentum)

        return optimizer

    def train_speech_net(self):
        """
        Train the model, and calculate the loss on the validtion data set , save the model if the values were improved
        check it by : validation loss and the mistake ratio
        """
        iteration = 0

        epoch = self.epoch
        optimizer = self.optimizer
        best_no_object_object_wrong_ratio = self.best_no_object_object_wrong_ratio
        best_correct_ratio = self.best_correct_ratio
        best_valid_loss = self.best_valid_loss

        # training with early stopping
        while (epoch < self.args.epochs + 1) and (iteration < self.args.patience):

            if epoch < 5 and self.args.opt == 'sgd':
                optimizer = optim.Adam(self.speech_net.parameters(), lr=self.args.lr)
            elif epoch == 5 and self.args.opt == 'sgd':
                optimizer = optim.SGD(self.speech_net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

            TrainSpeechYolo.train(self.train_loader, self.speech_net, self.loss.loss, self.config_dict, optimizer,
                                  epoch,
                                  self.args.cuda, self.args.log_interval)
            print(f"train epoch {epoch} ")

            valid_loss, six_part_loss, correct_ratio, no_object_object_wrong_ratio = TrainSpeechYolo.test(
                self.val_loader,
                self.speech_net,
                self.loss.loss,
                self.config_dict,
                self.args.decision_threshold,
                self.args.iou_threshold,
                self.args.cuda,
                print_progress=True)
            print('measures')
            threshold = 0.5
            TrainSpeechYolo.evaluation_measures(self.val_loader, self.speech_net, threshold, self.config_dict,
                                                self.args.cuda)

            if valid_loss > best_valid_loss and no_object_object_wrong_ratio > best_no_object_object_wrong_ratio:
                iteration += 1
                print('Loss was not improved, iteration {0}'.format(str(iteration)))
            else:
                print('Saving model...')
                iteration = 0
                best_correct_ratio = correct_ratio
                if best_valid_loss > valid_loss:
                    print("save best valid loss")
                    best_valid_loss = valid_loss
                if best_no_object_object_wrong_ratio >= no_object_object_wrong_ratio:
                    print("save best mistake")
                    best_no_object_object_wrong_ratio = no_object_object_wrong_ratio
                state = {
                    'net': self.speech_net.module.state_dict() if self.args.cuda else self.speech_net.state_dict(),
                    'acc': valid_loss,
                    'epoch': epoch,
                    'config_dict': self.config_dict,
                    'correct_ratio': best_correct_ratio,
                    'no_object_object_wrong_ratio': no_object_object_wrong_ratio,
                    'arc': self.args.arc,
                    'dropout': self.args.dropout,
                    'loss_params': {"noobject_conf": self.args.noobject_conf, "obj_conf": self.args.obj_conf,
                                    "coordinate": self.args.coordinate, "class_conf": self.args.class_conf,
                                    "loss_type": self.args.loss_type}
                }
                if not os.path.isdir(self.args.save_folder):
                    os.mkdir(self.args.save_folder)

                if self.args.save_file:
                    torch.save(state, self.args.save_folder + '/' + self.args.save_file)
                else:
                    torch.save(state, self.args.save_folder + '/' + build_model_name(self.args))
            epoch += 1

        self.epoch = epoch
        self.best_no_object_object_wrong_ratio = best_no_object_object_wrong_ratio
        self.best_correct_ratio = best_correct_ratio
        self.best_valid_loss = best_valid_loss

    def test_speech_net(self, folder_data_test, num_workers=0):
        """
        Test the speech net model
        :param folder_data_test: folder of the files to test
        :param num_workers: The number of workers that will be used to train the model and get the data,
                            you can change it by your machine capabilities

        The function test the model and print the accuracy values on the test data set
        """
        test_dataset = Datasets.SpeechYoloDataSet(classes_root_dir=self.args.train_data,
                                                  this_root_dir=folder_data_test,
                                                  yolo_config=self.config_dict)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.args.batch_size, shuffle=None,
            num_workers=num_workers, pin_memory=self.args.cuda, sampler=None)

        TrainSpeechYolo.test(test_loader, self.speech_net, self.loss.loss, self.config_dict,
                             self.args.decision_threshold,
                             self.args.iou_threshold, self.args.cuda, print_progress=True)


def main():
    data_folder_train = 'data/train'
    data_folder_valid = 'data/valid'
    data_folder_test = 'data/test'

    net = SpeechYoloVowels(data_folder_train, data_folder_valid, num_workers=8)
    net.train_speech_net()
    net.test_speech_net(data_folder_test, num_workers=8)


if __name__ == '__main__':
    main()
