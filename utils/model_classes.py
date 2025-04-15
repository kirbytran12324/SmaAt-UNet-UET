from models import unet_precip_regression_lightning as unet_regr
import lightning.pytorch as pl


def get_model_class(model_file) -> tuple[type[pl.LightningModule], str]:
    """
        This function takes a string representing a model file and returns a tuple containing
        the corresponding model class and a string representing the model's name.

        Parameters:
        model_file (str): A string representing the model file.

        Returns:
        tuple[type[pl.LightningModule], str]: A tuple containing the model class and the model's name.

        Raises:
        NotImplementedError: If the model name is not found in the model_file string.
    """
    if "UNet_Attention" in model_file:
        model_name = "UNet Attention"
        model = unet_regr.UNet_Attention
    elif "UNetDS_Attention_4kpl" in model_file:
        model_name = "UNetDS Attention with 4kpl"
        model = unet_regr.UNetDS_Attention
    elif "UNetDS_Attention_1kpl" in model_file:
        model_name = "UNetDS Attention with 1kpl"
        model = unet_regr.UNetDS_Attention
    elif "UNetDS_Attention_4CBAMs" in model_file:
        model_name = "UNetDS Attention 4CBAMs"
        model = unet_regr.UNetDS_Attention_4CBAMs
    elif "UNetDS_Attention" in model_file:
        model_name = "SmaAt-UNet"
        model = unet_regr.UNetDS_Attention
    elif "UNetDS" in model_file:
        model_name = "UNetDS"
        model = unet_regr.UNetDS
    elif "UNet" in model_file:
        model_name = "UNet"
        model = unet_regr.UNet
    else:
        raise NotImplementedError("Model not found")
    return model, model_name
