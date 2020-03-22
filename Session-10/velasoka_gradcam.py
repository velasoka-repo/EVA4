import torch.nn as nn
import torch
import numpy as np
import cv2

'''
    To support GradCam feature, I have refactored ResNet code a bit. 
    But didn't change any convolution layers | parameters etc...
'''


class GradientExtractor:
    def __init__(self):
        super(GradientExtractor, self).__init__()
        self.model = None
        self.gradients = None

    def __call__(self, gradient):
        self.gradients = gradient

    def get_gradient(self):
        return self.gradients


class ConvFeature:
    def __init__(self):
        super(ConvFeature, self).__init__()
        self.features = None

    def set_features(self, features):
        self.features = features.detach()  # detach features from gradient calculation

    def get_features(self):
        return self.features


class GradCamHook(nn.Module):
    def __init__(self, model, gradient_extractor, conv_feature):
        super(GradCamHook, self).__init__()
        self.model = model
        self.grad_extractor = gradient_extractor
        self.conv_feature = conv_feature

    def forward(self, x):
        out = self.model.feature_extractor(x)
        out.register_hook(self.grad_extractor)
        self.conv_feature.set_features(out)
        out = self.model.predict_layer(out)
        return out


class GradeCam():
    def __init__(self, grad_extractor, conv_feature, output):
        self.output = output.cpu()
        self.grad_extractor = grad_extractor
        self.conv_feature = conv_feature
        self.heatmap = None

    def get_predicted(self):
        """
        :return: predicted output
        """
        return self.output.argmax(dim=1)

    def get_heatmap(self):
        """
        :return: gradient heatmap for where feature is found
        """

        if self.heatmap:
            return self.heatmap

        pred = self.get_predicted()

        # back-propagation to compute gradient
        self.output[:, pred.item()].backward()
        gradient = self.grad_extractor.get_gradient()

        # average the gradient accross channels/features
        pool_gradient = torch.mean(gradient, dim=[0, 2, 3])
        conv_out = self.conv_feature.get_features()

        for i in range(512):
            # weight the corresponding channels/feature with gradient
            conv_out[:, i, :, :] *= pool_gradient[i]

        # average the channels/features
        self.heatmap = torch.mean(conv_out, dim=1).squeeze()

        # ReLU
        x = self.heatmap > 0
        x = self.heatmap * torch.tensor(x)
        x = torch.pow(x, 2)
        self.heatmap = torch.sqrt(x)

        # normalize
        self.heatmap /= torch.max(self.heatmap)
        return self.heatmap

    def lookup_feature(self, np_arr):
        img = np_arr.astype(np.uint8)
        heatmap = self.get_heatmap().cpu()
        np_heatmap = heatmap.cpu().numpy()

        # resize heatmap equal to image size
        np_heatmap = cv2.resize(np_heatmap, (img.shape[1], img.shape[0]))
        np_heatmap = np.uint8(255 * np_heatmap)
        np_heatmap = cv2.applyColorMap(np_heatmap, cv2.COLORMAP_JET)

        # add heatmap on top of image
        final_img = np_heatmap * 0.4 + img
        cv2.imwrite(f'final_img.jpg', final_img)
        img = cv2.imread(f"final_img.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


def extract_gradcam_list(data_list, classes):
    """
    :param classes: CIFAR10 class labels
    :param data_list: combined list [(img, label),(img, label)),  ......]
    :return: formatted cifar10 class names and images
    """
    tmp = []
    for data in data_list:
        actual, predicted = data
        img, label = actual
        title = f"Actual Label `{classes[label]}`"
        tmp.append((img, title))
        pre_img, pred_label = predicted
        title = f"Predicted Label: `{classes[pred_label]}`"
        tmp.append((pre_img, title))
    return tmp


def get_gradcam_image(model, data_loader, grad_extractor, conv_feature, denormalize_fn, count, device):
    """
    :param device:
    :param model: model to apply gradcam
    :param data_loader: test data loader
    :param grad_extractor: grad extractor
    :param conv_feature: conv feature in model
    :param denormalize_fn: method to use denormalize
    :param count: number of mis-classified image
    :return: (right * count, wrong * count) predicted image
    """
    model.eval()
    right_prediction = []
    wrong_prediction = []
    for idx, (data, label) in enumerate(data_loader):
        data = data.to(device)
        output = model(data)
        gradcam = GradeCam(grad_extractor=grad_extractor, conv_feature=conv_feature, output=output)
        predit_label = gradcam.get_predicted()
        img = denormalize_fn(data)  # to apply heatmap on top of this image, I am denormalizing the tensor
        visualize_img = gradcam.lookup_feature(img)

        tmp = ((img, label), (visualize_img, predit_label))

        if predit_label.item() == label.item() and len(right_prediction) < count:
            right_prediction += [tmp]
        elif predit_label.item() != label.item() and len(wrong_prediction) < count:
            wrong_prediction += [tmp]

        if len(right_prediction) == count and len(wrong_prediction) == count:
            break  # break test_dataloader loop

    return right_prediction, wrong_prediction
