import velasoka as vs
import velasoka_albumentations as al
import velasoka_model as model
import velasoka_gradcam as gr
import velasoka_lrfinder as lr


def view_augmentation_images(count=20):
    """
    show sample image augmentation using albumentation transforms
    """
    data, labels = al.read_cifar10_train_data()
    train_dataset = al.Dataset(data=data, labels=labels)

    # mean & std for train dataset
    mean, std = vs.mean_and_std(train_dataset.data)

    # pick random images from train dataset
    images, labels = vs.random_images(dataset=train_dataset, count=count)
    transform_list = al.data_transform_list(mean=mean)

    data_list = []
    for idx, image in enumerate(images):
        title = "" if idx != 0 else "Original"
        data_list.append((image, title))
        for name, fn in transform_list:
            title = "" if idx != 0 else name
            aug_image = al.augment_image(fn, image)
            data_list.append((aug_image, title))

    # display augmentation images
    vs.show_images(data_list=data_list, rows=len(images), columns=len(transform_list) + 1)


def trained_model(filename, folder):
    """
    :return: trained ResNet18 model
    """
    device = vs.available_device()

    TRAINED_MODEL_PATH = "resnet18-album-22.pt"
    if filename and folder:
        TRAINED_MODEL_PATH = vs.drive_path(filename=filename, folder=folder)

    # download CIFAR10 dataset
    vs.cifar10_dataset(transform=None, train=True)

    # load model
    resnet18 = model.ResNet18().to(device=device)
    stat_dict = vs.load_model(path=TRAINED_MODEL_PATH)
    resnet18.load_state_dict(state_dict=stat_dict)
    return resnet18


def test_accuracy_from_trained_model(model):
    """
    run Test dataloader to find Test Validation Accuracy
    """
    device = vs.available_device()

    # load model
    resnet18 = model.to(device)

    test_transform = vs.compose_transform(vs.to_tensor(), vs.to_normalize())
    test_dataset = vs.cifar10_dataset(transform=test_transform, train=False)
    test_dataloader = vs.cifar10_dataloader(dataset=test_dataset, batch_size=4)

    # Test Accuracy
    return vs.test_cnn(model=resnet18, data_loader=test_dataloader, device=device)


def best_lr_finder():
    test_transform = vs.compose_transform(vs.to_tensor(), vs.to_normalize())
    test_dataset = vs.cifar10_dataset(transform=test_transform, train=False)
    test_dataloader = vs.cifar10_dataloader(dataset=test_dataset, batch_size=4)

    train_transform = al.compose(al.normalize(), al.tensor())
    data, labels = al.read_cifar10_train_data()
    train_dataset = al.Dataset(data=data, labels=labels, transform=train_transform)
    train_dataloader = vs.cifar10_dataloader(dataset=train_dataset, batch_size=4, shuffle=True)

    device = vs.available_device()
    resnet18 = model.ResNet18().to(device=device)
    optimizer = vs.sgd_optimizer(model=resnet18, momentum=0)
    loss_fn = vs.ce_loss()

    # train lr finder
    print("-" * 150)
    print("Training Learning Rate Graph".rjust(70, ' '))
    print("-" * 150)
    lr_finder = lr.finder(model=resnet18, optimizer=optimizer, loss_fn=loss_fn, device=device)
    lr.train_range_test(lr_finder=lr_finder, train_loader=train_dataloader)
    lr.plot_loss(lr_finder=lr_finder)
    lr.reset(lr_finder=lr_finder)

    # test lr finder
    print("-" * 150)
    print("Training & Testing Learning Rate Graph".rjust(70, ' '))
    print("-" * 150)
    lr_finder = lr.finder(model=resnet18, optimizer=optimizer, loss_fn=loss_fn, device=device)
    lr.test_range_test(lr_finder=lr_finder, train_loader=train_dataloader, test_loader=test_dataloader)
    lr.plot_loss(lr_finder=lr_finder)
    lr.reset(lr_finder=lr_finder)
    print("-" * 150)
    print("Learning Rate History".rjust(70, ' '))
    print("-" * 150)
    history = lr.history(lr_finder=lr_finder)
    print(f"list of tried lr's\n{history}")


def gradcam_result(model, count=28):
    resnet18 = model
    optimizer = vs.sgd_optimizer(model=resnet18, lr=0.001, momentum=0.9)
    test_transform = vs.compose_transform(vs.to_tensor(), vs.to_normalize())
    test_dataset = vs.cifar10_dataset(transform=test_transform, train=False)
    test_dataloader = vs.cifar10_dataloader(dataset=test_dataset, batch_size=1)

    grad_extractor = gr.GradientExtractor()
    conv_feature = gr.ConvFeature()
    gradcamhook = gr.GradCamHook(model=resnet18, gradient_extractor=grad_extractor, conv_feature=conv_feature)
    device = vs.available_device()
    gradcamhook = gradcamhook.to(device)

    correct, wrong = gr.get_gradcam_image(model=gradcamhook, data_loader=test_dataloader, grad_extractor=grad_extractor,
                                          conv_feature=conv_feature, denormalize_fn=vs.denormalize_tensor, count=count,
                                          device=device)

    classes = al.cifar10_class_names()
    correct_list = gr.extract_gradcam_list(data_list=correct, classes=classes)
    wrong_list = gr.extract_gradcam_list(data_list=wrong, classes=classes)

    return correct_list, wrong_list


# TODO
def train_with_cutout_and_scheduler(model, epochs, train_loader):
    device = vs.available_device()
    test_transform = vs.compose_transform(vs.to_tensor(), vs.to_normalize())
    test_dataset = vs.cifar10_dataset(transform=test_transform, train=False)
    test_dataloader = vs.cifar10_dataloader(dataset=test_dataset, batch_size=4)

    train_transform = al.compose(al.normalize(), al.tensor())
    data, labels = al.read_cifar10_train_data()
    train_dataset1 = al.Dataset(data=data, labels=labels, transform=train_transform)
    train_dataloader1 = vs.cifar10_dataloader(dataset=train_dataset1, batch_size=4, shuffle=True)

    mean, std = al.mean_std(data)
    cutout_trans = al.cutout(fill_value=mean, p=1)
    train_dataset2 = al.transform_dataset(data, labels, cutout_trans, al.normalize(), al.tensor())
    train_dataloader2 = vs.cifar10_dataloader(dataset=train_dataset2, batch_size=4, shuffle=True)

    resnet18 = model
    optimizer = vs.sgd_optimizer(model=resnet18)
    loss_fn = vs.ce_loss()
    reduce_lr_scheduler = vs.reduce_lr_scheduler(optimizer=optimizer)

    norm_train_accuracy = []
    cutout_train_accuracy = []
    test_accuracy = []
    for epoch in range(1, epochs + 1):
        print(f"\nEPCOH: {epoch}")
        train_accu = vs.train_cnn(model=resnet18, data_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer,
                                  device=device)
        norm_train_accuracy.append(train_accu)
        train_accu = vs.train_cnn(model=resnet18, data_loader=train_dataloader2, loss_fn=loss_fn, optimizer=optimizer,
                                  device=device)
        cutout_train_accuracy.append(train_accu)
        test_accu = vs.test_cnn(model=resnet18, data_loader=test_dataloader, device=device)
        test_accuracy.append(test_accu)
        reduce_lr_scheduler.step(test_accu)  # step to reduce lr, if accuracy is less than previous

    vs.plot_accuracy_or_loss(norm_train_accuracy, "Normal Train dataset Accuracy")
    vs.plot_accuracy_or_loss(cutout_train_accuracy, "Cutout Train dataset Accuracy")
    vs.plot_accuracy_or_loss(test_accuracy, "Normal Test dataset Accuracy")