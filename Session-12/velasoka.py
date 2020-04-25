import utils.data as data
import utils.augment as aug
import utils.transform as tf
import model.nn as nn
import model.nn2 as nn2
import utils.visualize as view
import utils.torch_util as util
import utils.runner as network
import os

if __name__ == '__main__':
    working_dir = os.getcwd()
    # print(working_dir)
    data_path = f"../tiny-imagenet-200"
    print(data_path)

    # read tinyimagenet data
    img_dataset = data.LoadImage(root_path=data_path)
    mean, std = util.normalized_mean_std(img_dataset.mean, img_dataset.std)

    # create train & test dataset
    tf_util = tf.torch_transform()
    compose_transforms = tf.compose_transforms(tf_util.ToTensor(), tf_util.Normalize(mean=mean, std=std))
    train_dataset = data.ImageDataset(dataset=img_dataset.train_data, labels=img_dataset.train_labels,
                                      transforms=compose_transforms)
    test_dataset = data.ImageDataset(dataset=img_dataset.test_data, labels=img_dataset.test_labels,
                                     transforms=compose_transforms)

    # create train & test data loader
    kwargs = {'pin_memory': True, 'num_workers': 4}
    train_loader = data.TinyImageNetDataLoader(dataset=train_dataset, batch_size=10, shuffle=True, **kwargs).dataloader
    test_loader = data.TinyImageNetDataLoader(dataset=test_dataset, batch_size=10, shuffle=True, **kwargs).dataloader

    # create model & summary
    device = util.device()
    model = nn2.resnet18(num_classes=200).to(device)
    util.model_summary(model=model, input_size=(3, 64, 64))

    # if 1==1:
    #     raise Exception("done")
    trained_model_path = ""
    # trained_model_path = util.drive_path(filename="tiny-imagenet-resnet18.pt", folder="Session-12")
    if trained_model_path:
        state_dict = util.load_model(path=trained_model_path)
        model.load_state_dict(state_dict=state_dict)

    # create optimizer, loss fn, scheduler
    criterion = util.loss_fns().CrossEntropyLoss()
    sgd_optimizer = util.optimizer().SGD(params=model.parameters(), lr=0.01, momentum=0.9)
    scheduler = util.scheduler().ReduceLROnPlateau(optimizer=sgd_optimizer, mode="min", patience=5, verbose=True,
                                                   min_lr=0.00001, factor=0.1)

    # train & test
    epochs = 50
    train_accuracy = []
    test_accuracy = []
    max_train_accuracy_so_far = 0
    max_test_accuracy_so_far = 0
    for epoch in range(1, epochs + 1):
        print(f"EPOCH: {epoch}\n")
        train_acc = network.train(model=model, data_loader=train_loader, optimizer=sgd_optimizer, criterion=criterion,
                                  device=device)
        test_acc = network.test(model=model, data_loader=test_loader, device=device)
        scheduler.step(100 - test_acc)
        if max_train_accuracy_so_far < train_acc:
            max_train_accuracy_so_far = train_acc
            print(f"***Train Accuracy is greater than previous max accuracy. Saving model on {epoch}th epoch")
            util.save_model(model=model, path=trained_model_path)
    # print(img_dataset)
