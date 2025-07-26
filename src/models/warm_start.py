import torch

def warm_start_classification_model(regression_state, classification_model, shrink=0.9, perturb=0.01):
    """
    Transfers knowledge from regression model to classification model
    using the shrink-perturb method
        --------
    Parameters:
        regression_state : str
            file path to the saved .pth file that is to be loaded
        classification_model : torch.model
            initialized (but not trained) torch classification model
        shrink : double (optional)
            the amount to shrink the regression weights by way of multiplication
        preturb : doulbe (optional)
            the amount to displace the regression weights by
    --------
    Returns:
        classification_model : torch.model
            initialized with the saved weights from the .pth file of the trained regression model
    """
    regression_state = torch.load(regression_state)
    classification_state = classification_model.state_dict()

    new_state = {}
    
    for key in classification_state:
        if key in regression_state and 'prediction_head' not in key:
            if classification_state[key].shape == regression_state[key].shape:
                if classification_state[key].is_floating_point():
                    new_state[key] = regression_state[key] * shrink + \
                                    torch.randn_like(classification_state[key]) * perturb
                else:
                    new_state[key] = regression_state[key]
            else:
                # Shape mismatch: keep original classification weights
                new_state[key] = classification_state[key]
        else:
            new_state[key] = classification_state[key]
    
    # Load the modified state dict
    classification_model.load_state_dict(new_state)
    
    classification_model.load_state_dict(classification_state)

    return classification_model

def load_saved_model(saved_state, model):
    """
    Loads saved model state
    --------
    Parameters:
        saved_state : str
            file path to the saved .pth file that is to be loaded
        model : torch.model
            initialized (but not trained) torch model
    --------
    Returns:
        model : torch.model
            initialized with the saved weights from the .pth file
    """
    
    loaded_state = torch.load(saved_state)
   
    # Load the modified state dict
    model.load_state_dict(loaded_state)

    return model