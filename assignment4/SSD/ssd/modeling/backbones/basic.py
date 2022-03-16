# import torch
from torch import nn
from typing import Tuple, List

class BasicModel(nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        filter_size = 3
        padding = 1
        stride=1

        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(image_channels, 32, filter_size, 1, padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, filter_size, 1, padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, output_channels[0], filter_size, 2, padding),
            nn.BatchNorm2d(output_channels[0]),
            nn.ReLU(),
        )

        self.additional_layers = nn.ModuleList([
            # Block 2
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[0], 128, filter_size, 1, padding),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, output_channels[1], filter_size, 2, padding),
                nn.BatchNorm2d(output_channels[1]),
                nn.ReLU()
            ),
            # Block 3
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[1], 256, filter_size, 1, padding),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                # nn.Conv2d(256, 256, filter_size, 1, padding),
                # nn.BatchNorm2d(256),
                # nn.ReLU(),
                nn.Conv2d(256, output_channels[2], filter_size, 2, padding),
                nn.BatchNorm2d(output_channels[2]),
                nn.ReLU(),
            ),
            # Block 4
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[2], 128, filter_size, 1, padding),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                # nn.Conv2d(128, 128, filter_size, 1, padding),
                # nn.BatchNorm2d(128),
                # nn.ReLU(),
                nn.Conv2d(128, output_channels[3], filter_size, 2, padding),
                nn.BatchNorm2d(output_channels[3]),
                nn.ReLU(),
            ),
            # Block 5
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[3], 128, filter_size, 1, padding),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                # nn.Conv2d(128, 128, filter_size, 1, padding),
                # nn.BatchNorm2d(128),
                # nn.ReLU(),
                nn.Conv2d(128, output_channels[4], filter_size, 2, padding),
                nn.BatchNorm2d(output_channels[4]),
                nn.ReLU(),
            ),
            # Block 6
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[4], 128, filter_size, 1, padding),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                # nn.Conv2d(128, 128, filter_size, 1, padding),
                # nn.BatchNorm2d(128),
                # nn.ReLU(),
                nn.Conv2d(128, output_channels[5], filter_size, 1, 0),
                nn.BatchNorm2d(output_channels[5]),
                nn.ReLU(),
            )
        ])


            # Block 3


            # Block 4
            # nn.ReLU(),
            # nn.Conv2d(output_channels[2], 128, 3, 1, 1),
            # nn.ReLU(),
            # nn.Conv2d(128, output_channels[3], 3, 2, 1),
            # nn.ReLU(),

            # Block 5
            # nn.ReLU(),
            # nn.Conv2d(output_channels[3], 128, 3, 1, 1),
            # nn.ReLU(),
            # nn.Conv2d(128, output_channels[4], 3, 2, 1),
            # nn.ReLU(),

            # Block 6
            # nn.ReLU(),
            # nn.Conv2d(output_channels[4], 128, 3, 1, 1),
            # nn.ReLU(),
            # nn.Conv2d(128, output_channels[5], 3, 1, 0),
            # nn.ReLU(),
        # )

        print(self)
    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        out = self.feature_extractor(x)
        out_features.append(out)

        for l in range(len(self.additional_layers)):
            out = self.additional_layers[l](out)
            out_features.append(out)

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)