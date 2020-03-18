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
        x = self.heatmap>0
        x = self.heatmap * torch.tensor(x)
        x = torch.pow(x, 2)
        self.heatmap = torch.sqrt(x)

        #normalize
        self.heatmap /= torch.max(self.heatmap)
        return self.heatmap

    def lookup_feature(self, np_arr):
        img = np_arr.astype(np.uint8)
        heatmap = self.get_heatmap().cpu()
        np_heatmap = heatmap.cpu().numpy()

        #resize heatmap equal to image size
        np_heatmap = cv2.resize(np_heatmap, (img.shape[1], img.shape[0]))
        np_heatmap = np.uint8(255 * np_heatmap)
        np_heatmap = cv2.applyColorMap(np_heatmap, cv2.COLORMAP_JET)

        #add heatmap on top of image
        final_img = np_heatmap * 0.4 + img
        cv2.imwrite(f'final_img.jpg', final_img)
        img = cv2.imread(f"final_img.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img