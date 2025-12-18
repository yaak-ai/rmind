from torch import nn

from rmind.components.timm_backbone import TimmBackbone


class NoDownSampleBackbone(TimmBackbone):
    def __init__(
        self,
        model_name: str = "resnet18d.ra2_in1k",
        *,
        freeze: bool | None = None,
        out_indices: list[int] | None = None,
        img_size: list[int] | None = None,
    ) -> None:
        super().__init__(
            model_name, freeze=freeze, out_indices=out_indices, img_size=img_size
        )
        no_channels = self.model.layer1[0].conv1.in_channels  # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue]
        # set the dimensionality of all other layers to be the same as for layer1
        for layer in list(self.model.children())[-3:]:
            layer[0].conv1.stride = (1, 1)  # pyright: ignore[reportIndexIssue]

            for block in layer.modules():
                match block:
                    case nn.Conv2d():
                        block.in_channels = no_channels
                        block.out_channels = no_channels
                        block.weight = nn.Parameter(
                            block.weight[:no_channels, :no_channels, ...]
                        )
                        if block.bias is not None:
                            block.bias = nn.Parameter(block.bias[:no_channels])
                    case nn.BatchNorm2d():
                        block.num_features = no_channels
                        block.weight = nn.Parameter(block.weight[:no_channels])
                        block.bias = nn.Parameter(block.bias[:no_channels])
                        block.running_mean = nn.Parameter(
                            block.running_mean[:no_channels]  # pyright: ignore[reportOptionalSubscript]
                        )
                        block.running_var = nn.Parameter(
                            block.running_var[:no_channels]  # pyright: ignore[reportOptionalSubscript]
                        )
                    case _:
                        pass

            # Residual connection: remove downsampling of the input
            # Remove the (0): AvgPool2d(kernel_size=2, stride=2, padding=0)
            layer[0].downsample = layer[0].downsample[1:]  # pyright: ignore[reportIndexIssue]
