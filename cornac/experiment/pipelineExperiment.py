#############################################################################################################
# DESCRIPTION:  The `PipelineExperiment` class facilitates configurable recommendation experiments,         #
#               supporting model evaluation, reranking, and partial pipeline execution without retraining.  #
#############################################################################################################

import os
import configparser
import pickle
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm


from ..models.recommender import Recommender


from cornac.experiment.experiment import Experiment
from collections import OrderedDict
import json
import numpy as np
from ..eval_methods.static_rerank_evaluator import StaticReRankEval
from ..eval_methods.dynamic_rerank_evaluator import DynamicReRankEval


class PipelineExperiment(Experiment):
    """PipelineExperiment Class

    A flexible pipeline class for partial execution of an experiment.
    It allows skipping model training, and focuses on loading, reranking and evaluating results.

    Parameters
    ----------
    config_file: str, required
        The path to the .ini configuration file.

    eval_method: obj:`<cornac.eval_methods.BaseMethod>`, required
        The evaluation method (e.g., `RatioSplit`).

    rerankers: list of obj:`<cornac.rerankers.ReRanker>`, optional
        A list of rerankers that will be used to refine model recommendations.

    metrics: list of obj:`<cornac.metrics.RankingMetric>, <cornac.metrics.DiversityMetric>`, required
        A collection of metrics to use for evaluating the results.
    """

    def __init__(self,
                 
                 model,
                 metrics,
                eval_method = None,
                 rerankers=None,
                 user_based=True,
                 show_validation=True,
                 verbose=False,
                 pipeline_config_file=None):
        """
        Initializes the PipelineExperiment class, setting up models, metrics, rerankers, and configuration 
        for partial or complete execution of an experiment.

        Parameters:
        -----------
        models : Recommender or list of Recommender
            The recommender model(s) to evaluate. If a list is provided, only the first model will be used.

        metrics : list of cornac.metrics
            A collection of metrics to evaluate the model performance (e.g., RMSE, Precision, Recall).

        rerankers : list of ReRanker, optional
            A list of rerankers for refining recommendations from the base model.

        user_based : bool, optional (default=True)
            If True, evaluations are performed on a per-user basis.

        show_validation : bool, optional (default=True)
            Whether to display validation results during evaluation.

        verbose : bool, optional (default=False)
            If True, detailed logs and debug information will be printed.

        pipeline_config_file : str, optional
            Path to an .ini configuration file specifying pipeline parameters.

        Attributes:
        -----------
        config_file : str
            Path to the .ini configuration file.

        config : ConfigParser
            Parsed configuration file containing pipeline parameters.

        mode_and_paths : dict
            Modes and file paths for models, static rerankers, and dynamic rerankers extracted from the config.

        eval_method : cornac.eval_methods.BaseMethod
            Evaluation method used to split the dataset and compute metrics.

        save_dir : str
            Directory to save evaluation results and recommendations.

        models : Recommender
            The recommender model being evaluated.

        rerankers : list of ReRanker or None
            Static rerankers used for refining model recommendations.

        dynamic_rerankers : list of ReRanker or None
            Dynamic rerankers used for reordering recommendations based on user feedback.


        metrics : list of cornac.metrics
            Metrics used to evaluate the model's performance.

        user_based : bool
            Whether evaluations are performed on a per-user basis.

        show_validation : bool
            Whether validation results are displayed during evaluation.

        verbose : bool
            Whether detailed logs and debug information are printed.

        result : object or None
            The final test result of the experiment.

        val_result : object or None
            The validation result of the experiment.

        rerank_result : object or None
            The result after applying reranking strategies.

        Notes:
        ------
        This method validates the input models, metrics, and rerankers, loads configurations, and prepares
        the evaluation environment.
        """

        self.config_file = pipeline_config_file
        print(f"Pipeline experiment config file: {self.config_file}")
        self.config = self._parse_config(pipeline_config_file)
        self.mode_and_paths = self.get_mode_and_paths(self.config)
        if eval_method is None:
            self.eval_method = self.load_dataset(self.config)
        else:
            self.eval_method = eval_method 
        self.save_dir = self.config['pipeline'].get(
            'save_dir', '.')
        os.makedirs(self.save_dir, exist_ok=True)
        
        #  self.models is a `recommender`` object. This pipeline can only process one model.
        self.model = self._validate_models(model)
        # Validate and assign rerankers
        validated_rerankers = self._validate_rerankers(rerankers)

        self.rerankers = validated_rerankers.get('static', None) or None
        self.dynamic_rerankers = validated_rerankers.get(
            'dynamic', None) or None

        self.metrics = self._validate_metrics(metrics)
        self.user_based = user_based
        self.show_validation = show_validation
        self.verbose = verbose

        self.result = None
        self.val_result = None
        self.rerank_result = None


    @staticmethod
    def _validate_models(input_models):
        """
        Validates the input models and extracts a single Recommender object.

        Parameters:
        -----------
        input_models : Recommender or list of Recommender
            The input model(s) to validate.

        Returns:
        --------
        Recommender
            A single validated Recommender object.

        Raises:
        -------
        ValueError
            If the input is not a Recommender or a list of valid Recommender objects.
        """
        if isinstance(input_models, Recommender):
            return input_models  # return single object

        if hasattr(input_models, "__len__") and len(input_models) > 0:
            # Take only the first model from the list
            first_model = input_models[0]

            if isinstance(first_model, Recommender):
                return first_model
            else:
                raise ValueError(
                    f"The provided model is not a valid Recommender object: {type(first_model)}")
        else:
            raise ValueError(
                "input_models must be a Recommender or a list containing Recommender objects.")

    def _parse_config(self, config_file):
        """Parse the .ini config file.

        Parameters:
        -----------
        config_file : str
            Path to the configuration file.

        Returns:
        --------
        ConfigParser
            Parsed configuration object.

        Raises:
        -------
        ValueError
            If the file is empty or malformed.
        """
        config = configparser.ConfigParser()
        config.read(config_file)
        if not config.sections():
            raise ValueError(
                f"Config file {config_file} is empty or malformed.")
        return config

    def _parse_boolean(self, value):
        """
        Convert a string representation of a boolean into an actual boolean value.

        Parameters:
        -----------
        value : str
            String representing a boolean ('true', '1', 'yes', etc.).

        Returns:
        --------
        bool
            The corresponding boolean value.
        """
        return value.lower() in ('true', '1', 'yes')


    def load_dataset(self, config):
        """
        Load the dataset specified in the configuration file.

        Parameters:
        -----------
        config : ConfigParser
            The configuration object with the dataset path.

        Returns:
        --------
        cornac.eval_methods.BaseMethod
            The loaded dataset object.

        Raises:
        -------
        ValueError, FileNotFoundError
            If the dataset path is invalid or missing.
        """
        # from cornac.eval_methods.ratio_split import RatioSplit
        from ..eval_methods.base_method import BaseMethod
        """Extract the dataset path from the configuration."""
        try:
            # Check if the 'data' section exists in the config
            if 'data' not in config.sections():
                raise ValueError(
                    "'data' section is missing from the configuration.")

            # Get the dataset path from the 'data' section
            dataset_path = config.get('data', 'dataset_path', fallback=None)
            print(f"Loading eval_method from:{dataset_path}")
            if dataset_path is None:
                raise ValueError(
                    "dataset_path is not defined in the 'data' section.")

            # Validate if the dataset path exists in the file system
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(
                    f"Dataset path '{dataset_path}' does not exist.")

            dataset = BaseMethod.load(dataset_path)
            return dataset

        except Exception as e:
            print(f"Error extracting dataset path: {e}")
            return None

    def get_mode_and_paths(self, config):
        """
        Extract operational modes and file paths from the configuration file.

        Parameters:
        -----------
        config : ConfigParser
            The configuration object.

        Returns:
        --------
        dict
            A dictionary of modes and file paths for models and rerankers.

        Raises:
        -------
        ValueError
            If the 'pipeline' section is missing from the configuration.
        """
        if 'pipeline' not in config.sections():
            raise ValueError(
                "'pipeline' section is missing from the configuration.")

        # Extract mode and paths for model ranking
        model_mode = self._parse_boolean(
            config.get('pipeline', 'model_action'))
        model_ranked_items_path = config.get(
            'pipeline', 'model_ranked_items_path', fallback=None)
        model_eval_mode = self._parse_boolean(config.get(
            'pipeline', 'model_eval_action'))
        model_eval_result_path = config.get(
            'pipeline', 'model_eval_result_path', fallback=None)

        # Extract mode and paths for static reranking
        static_reranking_mode = self._parse_boolean(config.get(
            'pipeline', 'static_reranking_action'))
        static_reranked_items_path = config.get(
            'pipeline', 'static_reranked_items_path', fallback=None)
        static_eval_mode = self._parse_boolean(config.get
                                               ('pipeline', 'static_eval_action'))
        static_eval_result_path = config.get(
            'pipeline', 'static_eval_result_path', fallback=None)

        # Extract mode and paths for dynamic reranking
        dynamic_reranking_mode = self._parse_boolean(config.get(
            'pipeline', 'dynamic_reranking_action'))
        dynamic_reranked_items_path = config.get(
            'pipeline', 'dynamic_reranked_items_path', fallback=None)
        dynamic_eval_mode = self._parse_boolean(config.get(
            'pipeline', 'dynamic_eval_action'))
        dynamic_eval_result_path = config.get(
            'pipeline', 'dynamic_eval_result_path', fallback=None)
        
        parsed_result =  {
            'model': {'mode': model_mode, 'path': model_ranked_items_path, 'eval_mode': model_eval_mode, 'save_eval_path': model_eval_result_path},
            'static_reranker': {'mode': static_reranking_mode, 'path': static_reranked_items_path, 'eval_mode': static_eval_mode, 'save_eval_path': static_eval_result_path},
            'dynamic_reranker': {'mode': dynamic_reranking_mode, 'path': dynamic_reranked_items_path, 'eval_mode': dynamic_eval_mode, 'save_eval_path': dynamic_eval_result_path}
        }
        print(f"Configuration loaded:{parsed_result}")

        # Return a dictionary of options
        return parsed_result
    def load_model_recommendations(self, save_dir):
        """
        Load precomputed model recommendations from a pickle file.

        Parameters:
        -----------
        save_dir : str
            Path to the directory containing the recommendations file.

        Returns:
        --------
        dict
            Recommendations for each user.

        Raises:
        -------
        FileNotFoundError
            If the recommendations file is missing.
        """

        ranked_items_file_path = os.path.join(save_dir, "recommendations.pkl")

        if ranked_items_file_path and os.path.exists(ranked_items_file_path):
            print(
                f"Loading recommendations from {ranked_items_file_path}")

            with open(ranked_items_file_path, 'rb') as model_file:
                recommendation = pickle.load(model_file)
            return recommendation
        else:
            raise FileNotFoundError(
                f"No recommendation pkl file found at {ranked_items_file_path}. Please check the path and ensure the file exists.")

    def load_model_scores(self, save_dir):
        """
        Load precomputed item scores for all items from a pickle file.

        Parameters:
        -----------
        save_dir : str
            Path to the directory containing the scores file.

        Returns:
        --------
        dict
            Scores for each item.

        Raises:
        -------
        FileNotFoundError
            If the scores file is missing.
        """

        item_score_file_path = os.path.join(save_dir, "item_scores.pkl")
        mapped_indices_file_path = os.path.join(save_dir, "item_scores_mapped_indices.pkl")

        if item_score_file_path and os.path.exists(item_score_file_path):
            print(
                f"Loading scores from {item_score_file_path}")

            with open(item_score_file_path, 'rb') as model_file:
                item_scores = pickle.load(model_file)
            
        else:
            raise FileNotFoundError(
                f"No scores pkl file found at {item_score_file_path}. Please check the path and ensure the file exists.")

        # Load item_scores_mapped_indices (optional)
        if os.path.exists(mapped_indices_file_path):
            print(f"Loading item score mapped indices from {mapped_indices_file_path}")
            with open(mapped_indices_file_path, 'rb') as f:
                item_scores_mapped_indices = pickle.load(f)
        else:
            print(f"Warning: Item score mapped indices file not found at {mapped_indices_file_path}. Proceeding with default behavior: assuming item scores correspond to all available item indices in order [0, 1, ..., N].")
            item_scores_mapped_indices = {}

        return item_scores, item_scores_mapped_indices
    
    def save_results(self, test_result, val_result, save_dir, result_type="model"):
        """
        Save the results of the experiment to the specified directory.

        Parameters:
        -----------
        test_result : object
            The test result to save.
        val_result : object or None
            The validation result to save, if applicable.
        save_dir : str
            Directory to save the results.
        result_type : str, optional
            The type of result being saved (e.g., 'model', 'static_reranker'). Default is 'model'.

        """
        # Check if `all_test_results` attribute exists, if not create it as a dictionary
        if not hasattr(self, 'all_test_results'):
            self.all_test_results = {}

        # Add or update the test result for the given result type
        self.all_test_results[result_type] = test_result

        # If validation results need to be saved separately (optional)
        if not hasattr(self, 'all_val_results'):
            self.all_val_results = {}

        if val_result is not None:
            self.all_val_results[result_type] = val_result

        # Define the path to save the recommendation dictionary
        test_result.save(save_dir)

    def check_missing_recommendations(self, model, eval_method):
        """
        Checks for missing recommendations or prediction scores for each user.

        Parameters
        ----------
        model: object
            The model containing ranked_items and item_scores dictionaries.

        eval_method: `RatioSplit` object.
            The evaluation method containing the test set and rating threshold.

        Returns
        -------
        missing_user_indices: list
            A list of user indices for which recommendations or prediction scores are missing.

        Raises
        ------
        ValueError
            If there are any user indices with missing ranked items or prediction scores.
        """
        missing_user_indices = []

        def pos_items(csr_row):
            return [
                item_idx
                for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
                if rating >= eval_method.rating_threshold
            ]

        gt_mat = eval_method.test_set.csr_matrix
        test_user_indices = set(eval_method.test_set.uir_tuple[0])
        for user_idx in tqdm(
                test_user_indices, desc="Checking missing recommendations for users", miniters=100
        ):
            test_pos_items = pos_items(gt_mat.getrow(user_idx))
            if len(test_pos_items) == 0:
                continue

            # Check if ranked_items contains the user_idx and if it is a list or array
            if user_idx not in model.ranked_items or not isinstance(model.ranked_items[user_idx], (list, np.ndarray)):
                missing_user_indices.append(user_idx)

            # Check if item_scores contains the user_idx and if it is a list or array
            if user_idx not in model.item_scores or not isinstance(model.item_scores[user_idx], (list, np.ndarray)):
                missing_user_indices.append(user_idx)

        return missing_user_indices

    def run(self):
        """
        Execute the experiment pipeline, including model evaluation, reranking, and saving results.

        Raises:
        -------
        ValueError
            If recommendations or pipeline configurations are incomplete.
        """
        self._create_result()
        output = ""
        # for experiment, saving result purpose
        save_dir = self.save_dir
        if self.mode_and_paths["model"]['mode']:

            test_result, val_result = self.eval_method.evaluate(
                model=self.model,
                metrics=self.metrics,
                user_based=self.user_based,
                show_validation=self.show_validation
            )

            self.save_results(test_result, val_result,
                              self.mode_and_paths["model"]['save_eval_path'])
            self.model.save_recommendations(
                self.mode_and_paths["model"]['path'])
            output += "\n" + "="*8 + "model test result" + \
                "="*8 + "\n"+"{}".format(test_result)

        elif self.mode_and_paths["model"]['eval_mode']:
            save_dir = self.mode_and_paths["model"]['path']
            # load model recommendation result
            recommendations = self.load_model_recommendations(
                save_dir)
            self.model.ranked_items = recommendations

            self.model.item_scores, self.model.item_scores_mapped_indices = self.load_model_scores(save_dir)
            missing_user_indices = self.check_missing_recommendations(
                self.model, self.eval_method)
            # If there are any missing ranked items, raise an error with details
            if len(missing_user_indices) > 0:
                missing_users_str = ', '.join(str(uid)
                                              for uid in missing_user_indices)
                raise ValueError(
                    f"If skip training, the model should have ranked_items and item_scores for each user. Incomplete model ranked items or prediction scores found for user indices: {missing_users_str}"
                )

            test_result, val_result = self.eval_method.evaluate(
                model=self.model,
                metrics=self.metrics,
                user_based=self.user_based,
                show_validation=self.show_validation,
                train_mode=False
            )
            self.save_results(test_result, val_result,
                              self.mode_and_paths["model"]['save_eval_path'])

            output += "\n" + "="*8 + "model test result" + \
                "="*8 + "\n"+"{}".format(test_result)

        # load initial recommendation from the model. (The re-ranking process requires this data).
        else:

            save_dir = self.mode_and_paths["model"]['path']
            print(f"loading model {self.model.name} recommendation from: {save_dir}")
            recommendations = self.load_model_recommendations(
                save_dir)
            self.model.ranked_items = recommendations

            self.model.item_scores, self.model.item_scores_mapped_indices = self.load_model_scores(save_dir)

            # self.model.item_scores = self.load_model_scores(save_dir)

            # models.ranked_items must contain recommendation list for all user idx in the test_set!
            # check if the self.models.ranked_items ready.
            missing_user_indices = self.check_missing_recommendations(
                self.model, self.eval_method)
            # If there are any missing ranked items, raise an error with details
            if len(missing_user_indices) > 0:
                missing_users_str = ', '.join(str(uid)
                                              for uid in missing_user_indices)
                raise ValueError(
                    f"Incomplete model ranked items or prediction scores found for user indices: {missing_users_str}"
                )

        # If no missing items are found, print confirmation
        print("################# Start reranking ##########################")

        if self.mode_and_paths["static_reranker"]['mode']:
            static_reranker_evaluator = StaticReRankEval(self.eval_method)

            test_result_static_reranker, val_result_static_reranker = static_reranker_evaluator.evaluate(
                model=self.model, metrics=self.metrics,    user_based=self.user_based, rerankers=self.rerankers,  show_validation=self.show_validation)

            self.save_results(test_result_static_reranker, val_result_static_reranker,
                              self.mode_and_paths["static_reranker"]['save_eval_path'], result_type="static_reranker")

            output += "\n" + "="*8 + "static rerankers test result" + \
                "="*8 + "\n"+"{}".format(test_result_static_reranker)
            for reranker in self.rerankers:
                reranker_items_path = os.path.join(
                    self.mode_and_paths["static_reranker"]['path'], reranker.name)
                reranker.save_recommendations(reranker_items_path)
        elif self.mode_and_paths["static_reranker"]['eval_mode']:
            for reranker in self.rerankers:
                reranker_items_path = os.path.join(
                    self.mode_and_paths["static_reranker"]['path'], reranker.name)
                reranker.ranked_items = self.load_model_recommendations(
                    reranker_items_path)
            
            static_reranker_evaluator = StaticReRankEval(self.eval_method)

            test_result_static_reranker, val_result_static_reranker = static_reranker_evaluator.evaluate(
                model=self.model, metrics=self.metrics,    user_based=self.user_based, rerankers=self.rerankers,  show_validation=self.show_validation)
            self.save_results(test_result_static_reranker, val_result_static_reranker,
                              self.mode_and_paths["static_reranker"]['save_eval_path'], result_type="static_reranker")

            output += "\n" + "="*8 + "static rerankers test result" + \
                "="*8 + "\n"+"{}".format(test_result_static_reranker)
        if self.mode_and_paths["dynamic_reranker"]['mode']:
            dyn_reranker_evaluator = DynamicReRankEval(self.eval_method)
            test_result_dyn, val_result_dyn = dyn_reranker_evaluator.evaluate(
                model=self.model, metrics=self.metrics, user_based = False, rerankers=self.dynamic_rerankers, show_validation=self.show_validation)

            self.save_results(test_result_dyn, val_result_dyn,
                              self.mode_and_paths["dynamic_reranker"]['save_eval_path'], result_type="dynamic_reranker")

            output += "\n" + "="*8 + "dynamic rerankers test result" + \
                "="*8 + "\n" + "{}".format(test_result_dyn)
            for reranker in self.dynamic_rerankers:
                reranker_items_path = os.path.join(
                    self.mode_and_paths["dynamic_reranker"]['path'], reranker.name)
                reranker.save_recommendations(reranker_items_path)
        elif self.mode_and_paths["dynamic_reranker"]['eval_mode']:
            for reranker in self.dynamic_rerankers:
                reranker_items_path = os.path.join(
                    self.mode_and_paths["dynamic_reranker"]['path'], reranker.name)
                reranker.ranked_items = self.load_model_recommendations(
                    reranker_items_path)
                
            dyn_reranker_evaluator = DynamicReRankEval(self.eval_method)
            test_result_dyn, val_result_dyn = dyn_reranker_evaluator.evaluate(
                                model=self.model, metrics=self.metrics, user_based = False, rerankers=self.dynamic_rerankers, show_validation=self.show_validation)

            self.save_results(test_result_dyn, val_result_dyn,
                              self.mode_and_paths["dynamic_reranker"]['save_eval_path'], result_type="dynamic_reranker")

            output += "\n" + "="*8 + "dynamic rerankers test result" + \
                "="*8 + "\n" + "{}".format(test_result_dyn)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

        output_file = os.path.join(
            self.save_dir, "CornacExp-{}.log".format(timestamp))
        with open(output_file, "w") as f:
            f.write(output)
        print(output)
