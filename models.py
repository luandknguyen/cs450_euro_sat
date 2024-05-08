import torch
import torch.nn
import torch.nn.functional
from typing import Callable, Tuple, List, Any, Union

class EncoderBlock(torch.nn.Module):
    """Encoder block."""
    def __init__(self, in_channels, out_channels, max_pooling=True):
        super().__init__()
        self.max_pooling = max_pooling
        self.conv_1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.conv_2 = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        if max_pooling:
            self.max_pool = torch.nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            )

    def forward(self, inputs):
        """Fowarding."""
        matrix = torch.nn.functional.relu(self.conv_1(inputs))
        matrix = torch.nn.functional.relu(self.conv_2(matrix))
        skip = matrix
        if self.max_pooling:
            matrix = self.max_pool(matrix)
        return matrix, skip


class DecoderBlock(torch.nn.Module):
    """Decoder block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sampling = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv_1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding='same',
        )
        self.conv_2 = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding='same',
        )

    def forward(self, inputs, skip):
        """Forwarding."""
        matrix = self.up_sampling(inputs)
        matrix = torch.cat([matrix, skip], dim=1)
        matrix = torch.nn.functional.relu(self.conv_1(matrix))
        matrix = torch.nn.functional.relu(self.conv_2(matrix))
        return matrix


class Encoder(torch.nn.Module):
    """Encoder."""
    def __init__(self, input_channels: int, channels: List[int]):
        super().__init__()
        self.block_1 = EncoderBlock(in_channels=input_channels, out_channels=channels[0])
        self.block_2 = EncoderBlock(in_channels=channels[0], out_channels=channels[1])
        self.block_3 = EncoderBlock(in_channels=channels[1], out_channels=channels[2])
        self.block_4 = EncoderBlock(in_channels=channels[2], out_channels=channels[3], max_pooling=False)

    def forward(self, inputs):
        """Forwarding."""
        matrix, skip_1 = self.block_1(inputs)
        matrix, skip_2 = self.block_2(matrix)
        matrix, skip_3 = self.block_3(matrix)
        matrix, _ = self.block_4(matrix)
        return matrix, skip_1, skip_2, skip_3


class Decoder(torch.nn.Module):
    """Decoder."""
    def __init__(self, n_classes: int, channels: List[int]):
        super().__init__()
        self.block_1 = DecoderBlock(in_channels=channels[3] + channels[2], out_channels=channels[2])
        self.block_2 = DecoderBlock(in_channels=channels[2] + channels[1], out_channels=channels[1])
        self.block_3 = DecoderBlock(in_channels=channels[1] + channels[0], out_channels=channels[0])
        self.conv_out = torch.nn.Conv2d(
            in_channels=channels[0],
            out_channels=n_classes,
            kernel_size=3,
            padding='same',
        )
        self.softmax = torch.nn.Softmax2d()

    def forward(self, inputs, skip_1, skip_2, skip_3):
        """Forwarding."""
        matrix = self.block_1(inputs, skip_3)
        matrix = self.block_2(matrix, skip_2)
        matrix = self.block_3(matrix, skip_1)
        matrix = self.softmax(self.conv_out(matrix))
        return matrix


class UNet(torch.nn.Module):
    """U-Net"""
    def __init__(self, input_channels: int, n_classes: int, hidden_channels: List[int]):
        super().__init__()
        self.encoder = Encoder(input_channels=input_channels, channels=hidden_channels)
        self.decoder = Decoder(n_classes=n_classes, channels=hidden_channels)

    def forward(self, inputs):
        """Forwarding."""
        matrix, skip_1, skip_2, skip_3 = self.encoder(inputs)
        outputs = self.decoder(matrix, skip_1, skip_2, skip_3)
        return outputs


class Conv2Layers(torch.nn.Module):
    """Classifier."""
    def __init__(self, n_classes: int, image_size: Tuple[int, int], input_channels: int = 3):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(input_channels, 8, kernel_size=3, padding="same")
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding="same")
        self.pool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        in_channels = image_size[0]//4 * image_size[1]//4 * 16
        self.dense = torch.nn.Linear(in_channels, n_classes)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, inputs):
        matrix = torch.nn.functional.relu(self.conv_1(inputs))
        matrix = self.pool_1(matrix)
        matrix = torch.nn.functional.relu(self.conv_2(matrix))
        matrix = self.pool_2(matrix)
        matrix = torch.flatten(matrix, start_dim=1)
        matrix = self.softmax(self.dense(matrix))
        return matrix
