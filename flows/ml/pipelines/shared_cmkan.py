from . import usgs_unet


class SharedCmKANPipeline(usgs_unet.USGSUNetPipeline):
    """Paired spectral correction, without legacy checkpoint loading/freezing."""

    def __init__(
        self, model, spectral_channels: int = 31,
        use_ntire_colorimetry: bool = True, perceptual_weight: float = 0.1,
        **kwargs,
    ):
        if spectral_channels < 2:
            raise ValueError("Paired hyperspectral training requires at least two bands.")
        if use_ntire_colorimetry and spectral_channels != 31:
            raise ValueError("NTIRE colorimetry requires 31 bands at 400:10:700 nm.")
        if not use_ntire_colorimetry and perceptual_weight:
            raise ValueError("Disable perceptual_weight when use_ntire_colorimetry=false.")
        if kwargs.get("pretrain_epochs", 0) != 0:
            raise ValueError("Shared cmKAN trains paired correction; pretrain_epochs must be zero.")
        kwargs["pretrain_epochs"] = 0
        kwargs["masked_val_batches"] = 0
        super().__init__(model, perceptual_weight=perceptual_weight, **kwargs)
        self.spectral_channels = spectral_channels
        self.use_ntire_colorimetry = use_ntire_colorimetry
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x, y=None):
        if x.ndim != 4 or x.shape[1] != self.spectral_channels:
            raise ValueError(f"Expected {self.spectral_channels}-band BCHW input.")
        prediction = super().forward(x, y)
        if prediction.shape != x.shape:
            raise ValueError("Paired spectral correction must preserve the input cube shape.")
        return prediction

    def quality_metrics(self, pred, target):
        if self.use_ntire_colorimetry:
            return super().quality_metrics(pred, target)
        return self.spectral_quality_metrics(pred, target)
