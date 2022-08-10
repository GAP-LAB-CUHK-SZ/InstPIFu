import torch
import torch.nn as nn
import torch.nn.functional as F

def index(feat, uv):
    '''

    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    #uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True,mode='bilinear')  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]

class BasePIFuNet(nn.Module):
    def __init__(self,
                 projection_mode='orthogonal',
                 error_term=nn.L1Loss(),
                 ):
        """
        :param projection_mode:
        Either orthogonal or perspective.
        It will call the corresponding function for projection.
        :param error_term:
        nn Loss between the predicted [B, Res, N] and the label [B, Res, N]
        """
        super(BasePIFuNet, self).__init__()
        self.name = 'base'

        self.error_term = error_term

        self.index = index

        self.preds = None
        self.labels = None

    def forward(self, data_dict):
        '''
        :param points: [B, 3, N] world space coordinates of points
        :param images: [B, C, H, W] input images
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :return: [B, Res, N] predictions for each point
        '''
        images, points, intrinsic, labels = data_dict["image"], data_dict["sample_points"], data_dict["intrinsic"], \
                                            data_dict["gt_ndf"]
        transforms = None
        self.filter(images)

        # Phase 2: point query
        self.query(points=points, intrinsic=intrinsic, transforms=transforms, labels=labels)
        return self.get_preds()

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        None

    def query(self, points, intrinsic, transforms=None, labels=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        None

    def get_preds(self):
        '''
        Get the predictions from the last query
        :return: [B, Res, N] network prediction for the last query
        '''
        return self.preds

    def get_error(self):
        '''
        Get the network loss from the last query
        :return: loss term
        '''
        return self.error_term(self.preds, self.labels)
