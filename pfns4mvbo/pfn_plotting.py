import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pfns4mvbo.priors import cocabo_prior


def showPFNPosteriorDistributions(model_pfn,
                                  num_features=5,
                                  num_data=50):
    # we take a model, and create plots showing a test y value,
    # and the pfn's outputted posterior distribution for that
    # test point
    data = cocabo_prior.get_batch(1, num_data + 10, num_features, hyperparameters={'None': None})

    # casually partition the data
    x_train = data.x[10:]
    y_train = data.y[10:]
    x_test = data.x[:10]
    y_test = data.y[:10]

    # get pfn's output
    output = model_pfn(x_train, y_train, x_test).detach()
    output = torch.softmax(output, -1)
    output = (output / model_pfn.criterion.bucket_widths)[:, 0, :]

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))

    for i in range(2):
        for j in range(5):
            axes[i, j].plot(model_pfn.criterion.borders[1:] + model_pfn.criterion.bucket_widths / 2,
                            output[5 * i + j, :])
            axes[i, j].scatter(y_test[5 * i + j], 0)

    plt.tight_layout()

    return fig, axes


def showPFNvsCOCABOPosteriorDistributions(model_pfn,
                                          num_features=5,
                                          num_data=50):
    # we take a model, and create plots showing a test y value,
    # and the pfn's outputted posterior distribution for that
    # test point
    # first we gather data
    num_numeric = np.random.randint(0, num_features)
    num_nominal = num_features - num_numeric
    num_categories = np.random.randint(2, 11, num_nominal)
    input_space = cocabo_prior.get_input_space(num_features,
                                               num_numeric=num_numeric,
                                               num_nominal=num_nominal,
                                               num_categories=num_categories)
    data = cocabo_prior.get_batch(1, num_data + 10, num_features, hyperparameters={'None': None},
                                  **{'num_numeric': num_numeric,
                                     'num_nominal': num_nominal,
                                     'num_categories': num_categories})

    # casually partition the data
    x_train = data.x[10:]
    y_train = data.y[10:]
    x_test = data.x[:10]
    y_test = data.y[:10]

    # get pfn's output_pfn
    output_pfn = model_pfn(x_train, y_train, x_test).detach()
    output_pfn = torch.softmax(output_pfn, -1)
    output_pfn = (output_pfn / model_pfn.criterion.bucket_widths)[:, 0, :]

    # get cocabo output
    model_cocabo = cocabo_prior.get_model(input_space)
    _ = model_cocabo.fit(x_train[:, 0, :], y_train[:, 0, :])
    cocabo_mean, cocabo_variance = model_cocabo.predict(x_test[:, 0, :])

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
    x_axis_points = model_pfn.criterion.borders[1:] + model_pfn.criterion.bucket_widths / 2

    for i in range(2):
        for j in range(5):
            output_cocabo = scipy.stats.norm.pdf(x_axis_points, loc=cocabo_mean[5 * i + j].item(),
                                                 scale=(np.sqrt(cocabo_variance[5 * i + j].item())))
            axes[i, j].plot(x_axis_points, output_pfn[5 * i + j, :], label='PFN')
            axes[i, j].plot(x_axis_points, output_cocabo, label='CoCaBO')
            axes[i, j].scatter(y_test[5 * i + j], 0)
    axes[0, 0].legend()
    plt.tight_layout()

    return fig, axes


def showPFNvsCOCABO(model_pfn, num_data):
    num_features = 2
    num_numeric = 1
    num_nominal = 1
    num_categories = [2]
    input_space = cocabo_prior.get_input_space(num_features,
                                               num_numeric=num_numeric,
                                               num_nominal=num_nominal,
                                               num_categories=num_categories)
    data = cocabo_prior.get_batch(1, num_data, num_features, hyperparameters={'None': None},
                                  **{'num_numeric': num_numeric,
                                     'num_nominal': num_nominal,
                                     'num_categories': num_categories})

    # casually partition the data
    x_train = data.x[10:]
    y_train = data.y[10:]
    n_test_points = 100
    x_test = torch.vstack([
        torch.linspace(-0.5, 1.5, n_test_points).repeat(2),
        torch.repeat_interleave(torch.tensor([0,1]), n_test_points)
    ]).T.unsqueeze(1)


    output_pfn = model_pfn(x_train, y_train, x_test).detach()
    mean_pfn = model_pfn.criterion.mean(output_pfn).detach()
    lower_pfn = model_pfn.criterion.icdf(output_pfn, 0.025).flatten().detach()
    upper_pfn = model_pfn.criterion.icdf(output_pfn, 0.975).flatten().detach()

    # get cocabo output
    model_cocabo = cocabo_prior.get_model(input_space)
    _ = model_cocabo.fit(x_train[:, 0, :], y_train[:, 0, :])
    mean_cocabo, variance_cocabo = model_cocabo.predict(x_test[:, 0, :])
    lower_cocabo = (mean_cocabo - 1.96 * torch.sqrt(variance_cocabo)).detach().flatten()
    upper_cocabo = (mean_cocabo + 1.96 * torch.sqrt(variance_cocabo)).detach().flatten()

    # plotting time!
    if plt.get_fignums():
        plt.close()

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))

    colors = [(0, 0, 1),
              (0, 0.5, 1),
              (1, 0, 0),
              (1, 0.3, 0)]

    ax[0].plot(x_test[n_test_points:, 0, 0].detach(), mean_pfn[n_test_points:].flatten(),
             c=colors[0], label='PFN')
    ax[0].plot(x_test[:n_test_points, 0, 0].detach(), mean_pfn[:n_test_points].flatten(),
             c=colors[1], label='PFN')
    ax[0].fill_between(x_test[n_test_points:, 0, 0].detach(),
                       lower_pfn[n_test_points:],
                       upper_pfn[n_test_points:],
                       alpha=0.5, color=colors[0])
    ax[0].fill_between(x_test[:n_test_points, 0, 0].detach(),
                       lower_pfn[:n_test_points],
                       upper_pfn[:n_test_points],
                       alpha=0.5, color=colors[1])

    ax[1].plot(x_test[n_test_points:, 0, 0].detach(), mean_cocabo[n_test_points:].detach().flatten(),
               c=colors[2], label='CoCaBO')
    ax[1].plot(x_test[:n_test_points, 0, 0].detach(), mean_cocabo[:n_test_points].detach().flatten(),
               c=colors[3], label='CoCaBO')
    ax[1].fill_between(x_test[n_test_points:, 0, 0].detach(),
                       lower_cocabo[n_test_points:],
                       upper_cocabo[n_test_points:],
                       alpha=0.5, color=colors[2])
    ax[1].fill_between(x_test[:n_test_points, 0, 0].detach(),
                       lower_cocabo[:n_test_points],
                       upper_cocabo[:n_test_points],
                       alpha=0.5, color=colors[3])

    ax[0].scatter(x_train[:, 0, 0].detach(), y_train.detach(), c=x_train[:, 0, 1])
    ax[1].scatter(x_train[:, 0, 0].detach(), y_train.detach(), c=x_train[:, 0, 1])

    ax[2].plot(x_test[:n_test_points, 0, 0].detach(),
               torch.abs(mean_cocabo[n_test_points:] - mean_pfn[n_test_points:]).detach())
    ax[2].plot(x_test[:n_test_points, 0, 0].detach(),
               torch.abs(mean_cocabo[:n_test_points] - mean_pfn[:n_test_points]).detach())

    return fig, ax