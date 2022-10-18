## IMPORTS ##
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd

from scipy.optimize import minimize
from tqdm import tqdm
import pickle

# set up parallelization
from joblib import Parallel, delayed
import multiprocessing

# silence all warnings including runtime warnings
import warnings



# matching model
def matching_law_model(X, b, s, lmax):
    width = int(X[0])
    choice = X[1]
    reward = X[2]

    # get rewards for both odors with a moving window of width
    reward_0 = np.convolve(np.logical_and(reward == 1, choice == 0), np.ones(width), mode="full")[: len(reward)]
    reward_1 = np.convolve(np.logical_and(reward == 1, choice == 1), np.ones(width), mode="full")[: len(reward)]
    # determine reward ratio
    log_reward_ratio = np.nan_to_num(np.log(reward_1) - np.log(reward_0), posinf=lmax, neginf=-lmax, nan=0)
    # determine choice
    choice_prob = 1 / (1 + np.exp(-b - s * log_reward_ratio))
    return choice_prob


def negloglik_matching_law(x, choices, rewards, width):
    b = x[0]
    s = x[1]
    lmax = x[2]

    # get choice probabilities
    loglik = 0
    for i in range(len(choices)):
        choice_probs = matching_law_model([width, choices[i], rewards[i]], b, s, lmax)
        loglik += np.sum(np.log(choice_probs) * choices[i] + np.log(1 - choice_probs) * (1 - choices[i]))

    return -loglik


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    # number of cores to use
    num_cores = multiprocessing.cpu_count()

    # load data
    training_choice_set = np.loadtxt("../../../FlYMazeRL/data/mohanta2022/training_choice_set.csv", delimiter=",")
    training_reward_set = np.loadtxt("../../../FlYMazeRL/data/mohanta2022/training_reward_set.csv", delimiter=",")
    test_choice_set = np.loadtxt("../../../FlYMazeRL/data/mohanta2022/test_choice_set.csv", delimiter=",")
    test_reward_set = np.loadtxt("../../../FlYMazeRL/data/mohanta2022/test_reward_set.csv", delimiter=",")

    for width in tqdm([5, 10, 15, 30, 60]):
        model_name = "matching_law_{:d}".format(width)

        # optimize model in parallel

        # set up optimization
        def optimize_model(i):
            # sample with replacement
            indices = np.random.choice(len(training_choice_set), size=len(training_choice_set), replace=True)
            # fit model
            bootstrapped_choice_set = training_choice_set[indices]
            bootstrapped_reward_set = training_reward_set[indices]
            x0 = [np.random.exponential(1), np.random.exponential(1), np.random.exponential(1)]
            res = minimize(
                negloglik_matching_law,
                x0,
                args=(bootstrapped_choice_set, bootstrapped_reward_set, width),
                method="Nelder-Mead",
            )
            return res.x

        # run optimization
        n_bootstraps = 1000
        results = Parallel(n_jobs=num_cores)(delayed(optimize_model)(i) for i in range(n_bootstraps))

        parameters = np.array(results)
        parameters = pd.DataFrame(data=parameters, columns=["intercept", "slope", "lmax"])

        training_probs = []
        test_probs = []
        for i in tqdm(range(parameters.shape[0])):
            temp = [
                matching_law_model(
                    (width, training_choice_set[ind], training_reward_set[ind]),
                    parameters.iloc[i, 0],
                    parameters.iloc[i, 1],
                    parameters.iloc[i, 2],
                )
                for ind in range(len(training_choice_set))
            ]
            training_probs.append(temp)
            temp = [
                matching_law_model(
                    (width, test_choice_set[ind], test_reward_set[ind]),
                    parameters.iloc[i, 0],
                    parameters.iloc[i, 1],
                    parameters.iloc[i, 2],
                )
                for ind in range(len(test_choice_set))
            ]
            test_probs.append(temp)
        training_probs = np.array(training_probs)
        test_probs = np.array(test_probs)

        # calculate the normalized log likelihoods
        training_nll = np.exp(
            np.nanmean(
                training_choice_set * np.log(training_probs) + (1 - training_choice_set) * np.log(1 - training_probs),
                axis=2,
            )
        )
        test_nll = np.exp(
            np.nanmean(test_choice_set * np.log(test_probs) + (1 - test_choice_set) * np.log(1 - test_probs), axis=2)
        )

        fit_results = {
            "training_nll": training_nll,
            "test_nll": test_nll,
            "training_predictions": training_probs,
            "test_predictions": test_probs,
            "params": parameters,
        }

        with open(
            "../../FlYMazeRL_Fits/matching/mohanta2022/matching_law_{:d}_fit_results.pkl".format(width), "wb"
        ) as f:
            pickle.dump(fit_results, f)

