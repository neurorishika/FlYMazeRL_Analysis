import numpy as np
import pandas as pd
import arviz as az
import os
from tqdm import tqdm
from itertools import product

from flymazerl.agents.classical import *
from flymazerl.gym.environment import ymaze_baiting, ymaze_static
from flymazerl.utils import (
    generate_params_from_fits,
    get_schedule_histories,
    get_schedule_values,
    get_agent_value_history,
)
from flymazerl.utils import generate_random_schedule_with_blocks
import gym
import pickle

# supress depreciation warnings
import warnings

warnings.filterwarnings("ignore")

model_database = pd.read_csv("../../FlYMazeRL/model_description_mohanta.csv")

# filter to only acceptreject models
model_database = model_database[model_database["Variant"] == "acceptreject"]

# generate maps
model_name_map = {
    model_database.iloc[i]["SHORTCODE"]: model_database.iloc[i]["Model"] for i in range(len(model_database))
}
model_simple_abv_map = {
    model_database.iloc[i]["SHORTCODE"]: model_database.iloc[i]["ModelAbv"] for i in range(len(model_database))
}
model_class_name_map = {
    model_database.iloc[i]["AgentClass"]: model_database.iloc[i]["SHORTCODE"] for i in range(len(model_database))
}

# Load all fits
model_dir = "../../FlYMazeRL_Fits/acceptreject/mohanta2022/"
model_fits = {}
for i in os.listdir(model_dir):
    if i.endswith(".nc"):
        model_fits[i.split("_")[0]] = az.from_netcdf(os.path.join(model_dir, i))

# Load Training Data
training_choice_set = np.loadtxt("../../FlYMazeRL/data/mohanta2022/training_choice_set.csv", delimiter=",")
training_reward_set = np.loadtxt("../../FlYMazeRL/data/mohanta2022/training_reward_set.csv", delimiter=",")
# Load Test Data
test_choice_set = np.loadtxt("../../FlYMazeRL/data/mohanta2022/test_choice_set.csv", delimiter=",")
test_reward_set = np.loadtxt("../../FlYMazeRL/data/mohanta2022/test_reward_set.csv", delimiter=",")

model_classes = [eval(x) for x in model_class_name_map.keys()]

# calculate training and test likelihoods
n_bootstraps = 100


def get_probabilities(
    model, training_choice_set, training_reward_set, test_choice_set, test_reward_set, overwrite=True
):
    model_name = model.__name__
    model_abv = model_simple_abv_map[model_class_name_map[model_name]]

    # check if model has been fit and saved, if not fit
    if (
        not os.path.isfile(f"data/{model_abv}_probabilities.npz")
        and not os.path.isfile(f"data/{model_abv}_likelihoods.npz")
    ) or overwrite:

        training_probabilities = []
        test_probabilities = []

        training_nlikelihoods = []
        test_nlikelihoods = []

        # get model parameters
        parameters, policyparameters = generate_params_from_fits(
            model, n_bootstraps, sample_from_population=True, dataset="mohanta"
        )

        # loop over bootstraps
        for params, policyparams in tqdm(zip(parameters, policyparameters), total=n_bootstraps):
            print(params, policyparams)
            agent = model(ymaze_static(160), params, policyparams)

            # training data
            if (
                model_name != "HCQLearner_acceptreject"
                and model_name != "HQLearner_acceptreject"
                and model_name != "FHCQLearner_acceptreject"
                and model_name != "FHQLearner_acceptreject"
                and model_name != "DFHQLearner_acceptreject"
                and model_name != "DFHCQLearner_acceptreject"
            ):
                p_action = eval(
                    f"agent.vectorizedActionProbabilities({''.join([f'{p}={params[p]},' for p in params]+[f'{p}={policyparams[p]},' for p in policyparams])}actions_set=training_choice_set,rewards_set=training_reward_set).eval()"
                )
            else:
                p_action = eval(
                    f"agent.vectorizedActionProbabilities({''.join([f'{p}={params[p]},' for p in params]+[f'{p}={policyparams[p]},' for p in policyparams])}actions_set=training_choice_set,rewards_set=training_reward_set)[0].eval()"
                )
            training_p_action = p_action.reshape(-1, 160)

            # test data
            if (
                model_name != "HCQLearner_acceptreject"
                and model_name != "HQLearner_acceptreject"
                and model_name != "FHCQLearner_acceptreject"
                and model_name != "FHQLearner_acceptreject"
                and model_name != "DFHQLearner_acceptreject"
                and model_name != "DFHCQLearner_acceptreject"
            ):
                p_action = eval(
                    f"agent.vectorizedActionProbabilities({''.join([f'{p}={params[p]},' for p in params]+[f'{p}={policyparams[p]},' for p in policyparams])}actions_set=test_choice_set,rewards_set=test_reward_set).eval()"
                )
            else:
                p_action = eval(
                    f"agent.vectorizedActionProbabilities({''.join([f'{p}={params[p]},' for p in params]+[f'{p}={policyparams[p]},' for p in policyparams])}actions_set=test_choice_set,rewards_set=test_reward_set)[0].eval()"
                )
            test_p_action = p_action.reshape(-1, 160)

            # save probabilities
            training_probabilities.append(training_p_action)
            test_probabilities.append(test_p_action)

            # calculate normalized likelihoods
            training_likelihood = np.exp(
                np.nanmean(
                    training_choice_set * np.log(training_p_action)
                    + (1 - training_choice_set) * np.log(1 - training_p_action),
                    axis=1,
                )
            )
            test_likelihood = np.exp(
                np.nanmean(
                    test_choice_set * np.log(test_p_action) + (1 - test_choice_set) * np.log(1 - test_p_action), axis=1,
                )
            )

            # save normalized likelihoods
            training_nlikelihoods.append(training_likelihood)
            test_nlikelihoods.append(test_likelihood)

        training_probabilities = np.array(training_probabilities)
        test_probabilities = np.array(test_probabilities)

        training_nlikelihoods = np.array(training_nlikelihoods)
        test_nlikelihoods = np.array(test_nlikelihoods)

        # save as compressed numpy array
        np.savez_compressed(
            f"data/{model_abv}_probabilities.npz",
            training_probabilities=training_probabilities,
            test_probabilities=test_probabilities,
        )
        np.savez_compressed(
            f"data/{model_abv}_likelihoods.npz",
            training_nlikelihoods=training_nlikelihoods,
            test_nlikelihoods=test_nlikelihoods,
        )


import sys

index = int(sys.argv[1])
# ensure that the index is within the range of models
if index >= len(model_classes):
    raise ValueError(f"Index must be less than {len(model_classes)}")
model = model_classes[index]
get_probabilities(model, training_choice_set, training_reward_set, test_choice_set, test_reward_set)
