import math
import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt

from argparse import ArgumentParser

# loading data
import uproot

from canvasEX import Canvas

import scipy.special as ssp


FIT_PARS = ['p0','p1','p2']
FIT_RANGE = (400, 1500)

#-- parsing arguments
def parse_args():
    parser = ArgumentParser(description=__doc__)
    d = dict(help='%(default)s')
    parser.add_argument('input_file')
    parser.add_argument('signal_file', nargs='?')
    parser.add_argument('-e', '--output-file-extension', default='.pdf')
    parser.add_argument('-n', '--n-fits', type=int, default=5, **d)
    par_opts = parser.add_mutually_exclusive_group()
    par_opts.add_argument('-s', '--save-pars')
    par_opts.add_argument('-l', '--load-pars')

    parser.add_argument('-m', '--signal-events', type=float,
                          default=10000000, nargs='?', const=5)
    parser.add_argument('-g', "--guessTheMidPoints", type=bool, default=False)
    parser.add_argument('-t', "--halfPrediction", type=bool, default=False)
    parser.add_argument('-f', "--fix-hyperparams", type=bool, default=False)
    parser.add_argument('-b', "--bruteForceScan", type=bool, default=False)

    return parser.parse_args()
#--- data loading
def run():

    args = parse_args()
    fBkg=uproot.open(args.input_file)['DSJ75yStar03_TriggerJets_J75_yStar03_mjj_TLA2016binning']
    #fBkg=uproot.open(args.input_file)['DSJ100yStar06_TriggerJets_J100_yStar06_mjj_TLA2016binning']
    x_bkg, y_bkg, xerr_bkg, yerr_bkg= get_xy_pts(fBkg, FIT_RANGE)



#--Setting Train Data and Test Data
    ##---- extracting Training set
    #X_train=evenElements(x_bkg)
    #y_train=evenElements(y_bkg)
    #yerr_train=evenElements(yerr_bkg)
    ##---- extracting prediction set
    #X_test=oddElements(x_bkg)
    #y_test=oddElements(y_bkg)
    #yerr_test=oddElements(yerr_bkg)

    def formatChange(l):
        """changing a list to a np array with an extra dimension"""
        #l=np.array(l)
        #output=l[:, np.newaxis]
        output=torch.Tensor(l)
        return output


    def formatChangeX(l):
        """changing a list to a np array with an extra dimension"""
        #l=np.array(l)
        #output=l[:, np.newaxis]
        #print("outputX",output)
        output=torch.Tensor(l)
        output=output.unsqueeze(1)
        return output
    #---- extracting Training set
    #X_train=formatChange(evenElements(x_bkg))
    #y_train=formatChange(evenElements(y_bkg))
    # larger training set
    X_train=formatChangeX(x_bkg)
    y_train=formatChange(y_bkg)
    #yerr_train=formatChange(evenElements(yerr_bkg))
    #---- extracting prediction set
    X_test=formatChangeX(oddElements(x_bkg))
    y_test=formatChange(oddElements(y_bkg))
    #yerr_test=formatChange(oddElements(yerr_bkg))


    print("training set setup ending")

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=1)
    model = ExactGPModel(X_train, y_train, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
	{'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 500
    for i in range(training_iter):
	# Zero gradients from previous iteration
        optimizer.zero_grad()
	# Output from model
        output = model(X_train)
	# Calc loss and backprop gradients
        loss = -mll(output, y_train).sum()
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()
    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Initialize plots
    f, ((y1_ax, y2_ax), (y3_ax, y4_ax)) = plt.subplots(2, 2, figsize=(8, 8))
    # Test points every 0.02 in [0,1]

    # Make predictions
    with torch.no_grad():
        #test_x = torch.linspace(400,1200, 100).view(1, -1, 1).repeat(1, 1, 1)
        y_preds = likelihood(model(X_test))
	# Get mean
        mean = y_preds.mean
	# Get lower and upper confidence bounds
        lower, upper = y_preds.confidence_region()

	# Plot training data as black stars
        y1_ax.plot(train_x[0].detach().numpy(), train_y[0].detach().numpy(), 'k*')
        # Predictive mean as blue line
        y1_ax.plot(test_x[0].squeeze().numpy(), mean[0, :].numpy(), 'b')
        # Shade in confidence
        y1_ax.fill_between(test_x[0].squeeze().numpy(), lower[0, :].numpy(), upper[0, :].numpy(), alpha=0.5)
        y1_ax.set_ylim([1, 1e8])

        y1_ax.set_yscale('log')
        y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
        y1_ax.set_title('Observed Values (Likelihood)')
        y1_ax.savefig("figure.pdf")

        signif, _=res_significance(oddElements(y_train), mean)

        with Canvas("fit"+".eps") as can:
            can.ax.set_yscale('log')
            can.ax.errorbar(X_test, y_test,  fmt='.', color="red", label="testing points")
            can.ax.errorbar(X_train, y_train, fmt='.', color="k", label="training points")

            #can.ax.fill_between(t, mu - std, mu + std,
            #                facecolor=(0, 1, 0, 0.5),
            #                zorder=5, label=r'GP error = $1\sigma$')
            print("mean", mean)
            mean=[i*2 for i in mean.numpy().tolist()]
            print("X_test", X_test)
            print("y value", oddElements(y_train))
            print("mean: ", mean)
            can.ax.plot(X_test.squeeze(0).numpy(), np.asarray(mean), '-r', label="GP Prediction")

            can.ax.legend(framealpha=0)
            can.ax.set_ylabel('events')
            can.ratio.stem(X_test, signif, markerfmt='.', basefmt=' ')
            can.ratio.axhline(0, linewidth=1, alpha=0.5)
            can.ratio.set_xlabel(r'$m_{jj}$ [GeV]', ha='right', x=0.98)
            can.ratio.set_ylabel('significance')


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_size=1)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(batch_size=1), batch_size=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_xy_pts(f, x_range=None):
    vals, edges=f.numpy()
    #f.numpy()
    vals=np.asarray(vals)
    edges=np.array(edges)
    errors=np.sqrt(vals)
    center=(edges[:-1]+edges[1:])/2
    widths=np.diff(edges)

    if x_range is not None:
        low, high = x_range
        ok = (center > low) & (center < high)
    return center[ok], vals[ok], widths[ok], errors[ok]

def res_significance(Data, Bkg):
    pvals = []
    zvals = []
    chi2 = 0

    for i in range(len(Data)):
        nB = Bkg[i]*2
        nD= Data[i]
        print("nB:", nB)
        print("nD:", nD)
        if nD != 0:
            if nB > nD:
                pval = 1.-ssp.gammainc(nD+1.,nB)
            else:
                pval = ssp.gammainc(nD,nB)
            prob = 1-2*pval
            if prob > -1 and prob < 1:
                zval = math.sqrt(2.)*ssp.erfinv(prob)
            else:
                zval = np.inf
            if zval > 100: zval = 20
            if zval < 0: zval = 0
            if (nD < nB): zval = -zval
        else: zval = 0
        zvals.append(zval)
        if abs(nD)==0.:
            chi2+=2
        else:
            if ((nD - nB) ** 2 / abs(nD)) <2:
                chi2 += ((nD - nB) ** 2 / abs(nD))
            else :
                chi2 += 2

    zvals=np.array(zvals)
    return zvals, chi2

def oddElements(iList):
    return [iList[index] for index in range(len(iList)) if index%2==1]
def evenElements(iList):
    return [iList[index] for index in range(len(iList)) if index%2==0]


if __name__ == '__main__':
    run()
