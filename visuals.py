import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context='notebook', style='darkgrid',
        palette='bright', font='sans-serif',
        font_scale=2, color_codes=True)

def learning_curve(*models, **named_models):
    """
    Plot the empirical risk curves wrt the number of epochs for all the models inputed.
    The models can be passed as follows:
        learning_curve(model0, model1, name2=model2, name3=model3)
    where the `names` are display names.
    """
    plt.figure(figsize=(20,10))
    if models:
        for model in models:
            plt.plot(get_curve(model), linewidth=2.0, linestyle="-")

    if named_models:
        for name, model in named_models.items():
            plt.plot(get_curve(model), linewidth=2.0, linestyle="-", label=name)

        plt.grid(True)
        plt.legend(loc='best')
        plt.ylabel('Empirical Risk')
        plt.xlabel('Epochs')

    plt.show()


def get_curve(model):
    """
    Get the empirical risk values for each epoch
    """
    if not hasattr(model, "coef_"):
        raise Exception("The model hasn't been fitted yet.")

    w_tab = model._coef_tab
    Rn = model._empirical_risk

    curve = [Rn(w) for w in w_tab]
    return curve
