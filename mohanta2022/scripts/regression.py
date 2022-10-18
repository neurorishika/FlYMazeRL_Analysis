## IMPORTS ##
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
import pickle

from scipy.optimize import minimize
from tqdm import tqdm

# set up parallelization
from joblib import Parallel, delayed
import multiprocessing

from sklearn.linear_model import LogisticRegressionCV

# regression model (reward+choice+choice*reward)
def regression_model_rci(width, choices, rewards, train=True, model=None):
    xs = []
    for i in range(choices.shape[0]):
        choice = 2 * choices[i] - 1
        reward = rewards[i]
        # pad choices  and reward with 0 for width trials on either side
        choice = np.concatenate((np.zeros(width), choice, np.zeros(width)))
        reward = np.concatenate((np.zeros(width), reward, np.zeros(width)))
        # roll choices and rewards to get a matrix of choices and rewards for each trial
        choice_matrix = np.array([choice[i : i + width] for i in range(len(choice) - 2 * width)])
        reward_matrix = np.array([reward[i : i + width] for i in range(len(reward) - 2 * width)])
        # join choices and rewards
        x = np.concatenate((choice_matrix, reward_matrix, choice_matrix * reward_matrix), axis=1)
        xs.append(x)
    xs = np.concatenate(xs)
    if model is None:
        model = LogisticRegressionCV()
    if train:
        model.fit(xs, choices.flatten())
    predictions = model.predict_proba(xs)[:, 1].reshape(choices.shape)
    nll = np.exp(np.nanmean(choices * np.log(predictions) + (1 - choices) * np.log(1 - predictions), axis=1))
    return model, predictions, nll


# regression model (reward+choice)
def regression_model_rc(width, choices, rewards, train=True, model=None):
    xs = []
    for i in range(choices.shape[0]):
        choice = 2 * choices[i] - 1
        reward = rewards[i]
        # pad choices  and reward with 0 for width trials on either side
        choice = np.concatenate((np.zeros(width), choice, np.zeros(width)))
        reward = np.concatenate((np.zeros(width), reward, np.zeros(width)))
        # roll choices and rewards to get a matrix of choices and rewards for each trial
        choice_matrix = np.array([choice[i : i + width] for i in range(len(choice) - 2 * width)])
        reward_matrix = np.array([reward[i : i + width] for i in range(len(reward) - 2 * width)])
        # join choices and rewards
        x = np.concatenate((choice_matrix, reward_matrix), axis=1)
        xs.append(x)
    xs = np.concatenate(xs)
    if model is None:
        model = LogisticRegressionCV()
    if train:
        model.fit(xs, choices.flatten())
    predictions = model.predict_proba(xs)[:, 1].reshape(choices.shape)
    nll = np.exp(np.nanmean(choices * np.log(predictions) + (1 - choices) * np.log(1 - predictions), axis=1))
    return model, predictions, nll


# regression model (reward+interaction)
def regression_model_ri(width, choices, rewards, train=True, model=None):
    xs = []
    for i in range(choices.shape[0]):
        choice = 2 * choices[i] - 1
        reward = rewards[i]
        # pad choices  and reward with 0 for width trials on either side
        choice = np.concatenate((np.zeros(width), choice, np.zeros(width)))
        reward = np.concatenate((np.zeros(width), reward, np.zeros(width)))
        # roll choices and rewards to get a matrix of choices and rewards for each trial
        choice_matrix = np.array([choice[i : i + width] for i in range(len(choice) - 2 * width)])
        reward_matrix = np.array([reward[i : i + width] for i in range(len(reward) - 2 * width)])
        # join choices and rewards
        x = np.concatenate((reward_matrix, choice_matrix * reward_matrix), axis=1)
        xs.append(x)
    xs = np.concatenate(xs)
    if model is None:
        model = LogisticRegressionCV()
    if train:
        model.fit(xs, choices.flatten())
    predictions = model.predict_proba(xs)[:, 1].reshape(choices.shape)
    nll = np.exp(np.nanmean(choices * np.log(predictions) + (1 - choices) * np.log(1 - predictions), axis=1))
    return model, predictions, nll


# regression model (choice+interaction)
def regression_model_ci(width, choices, rewards, train=True, model=None):
    xs = []
    for i in range(choices.shape[0]):
        choice = 2 * choices[i] - 1
        reward = rewards[i]
        # pad choices  and reward with 0 for width trials on either side
        choice = np.concatenate((np.zeros(width), choice, np.zeros(width)))
        reward = np.concatenate((np.zeros(width), reward, np.zeros(width)))
        # roll choices and rewards to get a matrix of choices and rewards for each trial
        choice_matrix = np.array([choice[i : i + width] for i in range(len(choice) - 2 * width)])
        reward_matrix = np.array([reward[i : i + width] for i in range(len(reward) - 2 * width)])
        # join choices and rewards
        x = np.concatenate((choice_matrix, choice_matrix * reward_matrix), axis=1)
        xs.append(x)
    xs = np.concatenate(xs)
    if model is None:
        model = LogisticRegressionCV()
    if train:
        model.fit(xs, choices.flatten())
    predictions = model.predict_proba(xs)[:, 1].reshape(choices.shape)
    nll = np.exp(np.nanmean(choices * np.log(predictions) + (1 - choices) * np.log(1 - predictions), axis=1))
    return model, predictions, nll


def fit_regression_model(model_function, name):
    # number of cores to use
    num_cores = multiprocessing.cpu_count()

    # load data
    training_choice_set = np.loadtxt("../../../FlYMazeRL/data/mohanta2022/training_choice_set.csv", delimiter=",")
    test_choice_set = np.loadtxt("../../../FlYMazeRL/data/mohanta2022/test_choice_set.csv", delimiter=",")
    training_reward_set = np.loadtxt("../../../FlYMazeRL/data/mohanta2022/training_reward_set.csv", delimiter=",")
    test_reward_set = np.loadtxt("../../../FlYMazeRL/data/mohanta2022/test_reward_set.csv", delimiter=",")

    # optimize model in parallel
    for width in tqdm([5, 10, 15, 30, 60]):
        model_name = "{:s}_{:d}".format(name, width)
        # set up optimization
        def optimize_model(i):
            # sample with replacement
            indices = np.random.choice(len(training_choice_set), size=len(training_choice_set), replace=True)
            # fit model
            bootstrapped_choice_set = training_choice_set[indices]
            bootstrapped_reward_set = training_reward_set[indices]
            model, training_predictions, training_nll = model_function(
                width, bootstrapped_choice_set, bootstrapped_reward_set
            )
            # test model
            model, test_predictions, test_nll = model_function(
                width, test_choice_set, test_reward_set, train=False, model=model
            )
            return model, training_predictions, training_nll, test_predictions, test_nll

        # run optimization
        n_bootstraps = 1000
        results = Parallel(n_jobs=num_cores)(delayed(optimize_model)(i) for i in range(n_bootstraps))

        parameters = np.array([list(result[0].coef_[0]) + [result[0].intercept_[0]] for result in results])
        models = [result[0] for result in results]
        training_predictions = np.array([result[1] for result in results])
        training_nll = np.array([result[2] for result in results])
        test_predictions = np.array([result[3] for result in results])
        test_nll = np.array([result[4] for result in results])

        if model_name.startswith("regression_rci"):
            columns = (
                ["w_choice_{:d}".format(width - i) for i in range(width)]
                + ["w_reward_{:d}".format(width - i) for i in range(width)]
                + ["w_choice_reward_{:d}".format(width - i) for i in range(width)]
                + ["bias"]
            )
        elif model_name.startswith("regression_ri"):
            columns = (
                ["w_reward_{:d}".format(width - i) for i in range(width)]
                + ["w_choice_reward_{:d}".format(width - i) for i in range(width)]
                + ["bias"]
            )
        elif model_name.startswith("regression_ci"):
            columns = (
                ["w_choice_{:d}".format(width - i) for i in range(width)]
                + ["w_choice_reward_{:d}".format(width - i) for i in range(width)]
                + ["bias"]
            )
        else:
            columns = (
                ["w_choice_{:d}".format(width - i) for i in range(width)]
                + ["w_reward_{:d}".format(width - i) for i in range(width)]
                + ["bias"]
            )

        params = pd.DataFrame(data=parameters, columns=columns)
        # save models and predictions
        fit_results = {
            "models": models,
            "training_predictions": training_predictions,
            "training_nll": training_nll,
            "test_predictions": test_predictions,
            "test_nll": test_nll,
            "params": params,
        }
        with open("../../../FlYMazeRL_Fits/regression/mohanta2022/{:s}_fit_results.pkl".format(model_name), "wb") as f:
            pickle.dump(fit_results, f)


if __name__ == "__main__":
    # run model
    fit_regression_model(regression_model_rci, "regression_rci")
    fit_regression_model(regression_model_ci, "regression_ci")
    fit_regression_model(regression_model_rc, "regression_rc")
    fit_regression_model(regression_model_ri, "regression_ri")

