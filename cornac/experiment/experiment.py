# Copyright 2018 The Cornac Authors. All Rights Reserved.
############################################################################
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
from datetime import datetime

from .result import ExperimentResult
from .result import CVExperimentResult
from ..metrics.rating import RatingMetric
from ..metrics.ranking import RankingMetric
from ..metrics.diversity import DiversityMetric
from ..models.recommender import Recommender
from ..rerankers.reranker import ReRanker
from ..rerankers.dynamic_reranker import DynamicReRanker
from ..eval_methods.static_rerank_evaluator import StaticReRankEval
from ..eval_methods.dynamic_rerank_evaluator import DynamicReRankEval

import json
from collections import OrderedDict


class Experiment:
    """ Experiment Class

    Parameters
    ----------
    eval_method: :obj:`<cornac.eval_methods.BaseMethod>`, required
        The evaluation method (e.g., RatioSplit).

    models: array of :obj:`<cornac.models.Recommender>`, required
        A collection of recommender models to evaluate, e.g., [C2PF, HPF, PMF].

    rerankers : dict, optional
            A dictionary containing static and dynamic rerankers for refining recommendations. Example:
            {'static': [ReRanker()], 'dynamic': [DynamicReRanker()]}
    metrics: array of :obj:{E.g., `<cornac.metrics.RankingMetric>`}, required
        A collection of metrics to use to evaluate the recommender models, 
        e.g., [NDCG, MRR, Recall].

    user_based: bool, optional, default: True
        This parameter is only useful if you are considering rating metrics. When True, first the average performance \
        for every user is computed, then the obtained values are averaged to return the final result.
        If `False`, results will be averaged over the number of ratings.

    show_validation: bool, optional, default: True
        Whether to show the results on validation set (if exists).
    
    verbose: bool, optional, default: False
        Output running log/progress during model training and evaluation.
        If verbose is True, it will overwrite verbosity setting of evaluation method and models.
        
    save_dir: str, optional, default: None
        Path to a directory for storing trained models and logs. If None,
        models will NOT be stored and logs will be saved in the current working directory.

    Attributes
    ----------
    result: array of :obj:`<cornac.experiment.result.Result>`, default: None
        This attribute contains the results per-model of your experiment
        on the test set, initially it is set to None.

    val_result: array of :obj:`<cornac.experiment.result.Result>`, default: None
        This attribute contains the results per-model of your experiment
        on the validation set (if exists), initially it is set to None.

    """

    def __init__(
        self,
        eval_method,
        models,
        metrics,
        rerankers=None,
        user_based=True,
        show_validation=True,
        verbose=False,
        save_dir=None,
    ):
        self.eval_method = eval_method
        self.models = self._validate_models(models)
        validated_rerankers = self._validate_rerankers(rerankers)
        self.rerankers = validated_rerankers.get('static', None) or None
        self.dynamic_rerankers = validated_rerankers.get(
            'dynamic', None) or None
        self.metrics = self._validate_metrics(metrics)
        self.user_based = user_based
        self.show_validation = show_validation
        self.verbose = verbose
        self.save_dir = save_dir
        self.result = None
        self.static_reranking_result = None
        self.dynamic_reranking_result = None
        self.val_result = None

    @staticmethod
    def _validate_models(input_models):
        if not hasattr(input_models, "__len__"):
            raise ValueError(
                "models have to be an array but {}".format(type(input_models))
            )

        valid_models = []
        for model in input_models:
            if isinstance(model, Recommender):
                valid_models.append(model)
        return valid_models

    @staticmethod
    def _validate_rerankers(input_rerankers):
        """
        Validate the static and dynamic rerankers provided for the experiment.

        Parameters
        ----------
        input_rerankers : dict
            A dictionary containing 'static' and 'dynamic' keys with lists of rerankers.

        Returns
        -------
        valid_rerankers : dict
            A dictionary containing validated static and dynamic rerankers.

        Raises
        ------
        ValueError
            If the rerankers are not in the expected format or contain invalid instances.
        """

        if input_rerankers is None:
            return {'static': None, 'dynamic': None}

        # Check if input_rerankers is a dictionary
        if not isinstance(input_rerankers, dict):
            raise ValueError(
                "rerankers must be a dictionary with 'static' and 'dynamic' keys"
            )

        # Initialize valid_rerankers dictionary

        valid_rerankers = {'static': [], 'dynamic': []}

        # Validate static rerankers if provided
        if 'static' in input_rerankers:
            if not isinstance(input_rerankers['static'], list):
                raise ValueError("'static' rerankers must be a list")
            for reranker in input_rerankers['static']:
                if isinstance(reranker, ReRanker):
                    valid_rerankers['static'].append(reranker)
                else:
                    raise ValueError(
                        "All static rerankers must be instances of ReRanker")

        # Validate dynamic rerankers if provided
        if 'dynamic' in input_rerankers:
            if not isinstance(input_rerankers['dynamic'], list):
                raise ValueError("'dynamic' rerankers must be a list")
            for reranker in input_rerankers['dynamic']:
                if isinstance(reranker, DynamicReRanker):
                    valid_rerankers['dynamic'].append(reranker)
                else:
                    raise ValueError(
                        "All dynamic rerankers must be instances of DynamicReRanker")
        if len(valid_rerankers['static']) == 0:
            valid_rerankers['static'] = None
        if len(valid_rerankers['dynamic']) == 0:
            valid_rerankers['dynamic'] = None
        print(
            f"Validated static rerankers: {len(valid_rerankers['static']) if valid_rerankers['static'] else 0}")
        print(
            f"Validated dynamic rerankers: {len(valid_rerankers['dynamic']) if valid_rerankers['dynamic'] else 0}")

        return valid_rerankers

    def _validate_dyn_rerankers(self, input_dyn_rerankers):
        """
        Validate the list of dynamic rerankers.

        Parameters
        ----------
        input_dyn_rerankers : list
            A list of dynamic rerankers to validate.

        Returns
        -------
        valid_rerankers : list
            A list containing only valid instances of `DynamicReRanker`.

        Raises
        ------
        ValueError
            If `input_dyn_rerankers` is not an array or contains invalid reranker instances.
        """

        if input_dyn_rerankers is None:
            return None
        if not hasattr(input_dyn_rerankers, "__len__"):
            raise ValueError(
                "rerankers have to be an array but {}".format(
                    type(input_dyn_rerankers))
            )

        valid_rerankers = []
        for reranker in input_dyn_rerankers:
            if isinstance(reranker, DynamicReRanker):
                valid_rerankers.append(reranker)
        print("validated rerankers: {}".format(len(valid_rerankers)))
        return valid_rerankers

    @staticmethod
    def _validate_metrics(input_metrics):
        """
        Validate the metrics provided for evaluation.

        Parameters
        ----------
        input_metrics : list
            A list of metrics to validate.

        Returns
        -------
        valid_metrics : list
            A list containing valid metrics (instances of `RatingMetric`, `RankingMetric`, or `DiversityMetric`).

        Raises
        ------
        ValueError
            If `input_metrics` is not an array or contains invalid metric instances.
        """
        if input_metrics == None:
            return None
        if not hasattr(input_metrics, "__len__"):
            raise ValueError(
                "metrics have to be an array but {}".format(
                    type(input_metrics))
            )

        valid_metrics = []
        for metric in input_metrics:
            if isinstance(metric, RatingMetric) or isinstance(metric, RankingMetric) or isinstance(metric, DiversityMetric):
                valid_metrics.append(metric)
        return valid_metrics

    def _create_result(self):
        from ..eval_methods.cross_validation import CrossValidation
        from ..eval_methods.propensity_stratified_evaluation import (
            PropensityStratifiedEvaluation,
        )

        if isinstance(self.eval_method, CrossValidation) or isinstance(
            self.eval_method, PropensityStratifiedEvaluation
        ):
            self.result = CVExperimentResult()
        else:
            self.result = ExperimentResult()
            self.static_reranking_result = ExperimentResult() 
            self.dynamic_reranking_result = ExperimentResult()
            if self.show_validation and self.eval_method.val_set is not None:
                self.val_result = ExperimentResult()


    def run(self):
        """Run the Cornac experiment.  Applying reranking if specified."""
        self._create_result()

        save_dir = self.save_dir or '.'

        # Ensure the directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # overwrite verbosity setting of evaluation method and models
        # if Experiment verbose is True
        if self.verbose:
            self.eval_method.verbose = self.verbose
            for model in self.models:
                model.verbose = self.verbose
    
        for model in self.models:

            test_result, val_result = self.eval_method.evaluate(
                model=model,
                metrics=self.metrics,
                user_based=self.user_based,
                rerankers=self.rerankers,
                dynamic_rerankers=self.dynamic_rerankers,
                show_validation=self.show_validation
            )
            # print(f"test result of model:{test_result}")

            self.result.append(test_result)
            
            if self.val_result is not None:
                self.val_result.append(val_result)

            if save_dir and (not isinstance(self.result, CVExperimentResult)):
                saving_model_path = save_dir + '/'+model.name
                model.save_recommendations(saving_model_path)

            if self.rerankers is not None:
                static_reranker_evaluator = StaticReRankEval(self.eval_method)
                
                print(f"Static reranking evaluation for model {model.name} started")
                for j in range(len(self.rerankers)):
                    reranker = self.rerankers[j]
                    reranker.reset()

                test_result_static_reranker, val_result_static_reranker = static_reranker_evaluator.evaluate(
                    model = model, metrics=self.metrics, user_based = self.user_based, rerankers=self.rerankers,show_validation= self.show_validation)
                
                self.static_reranking_result.append(test_result_static_reranker)
            
                if save_dir and (not isinstance(self.static_reranking_result, CVExperimentResult)):
                    for reranker in self.rerankers:
                        saving_model_path = save_dir + '/'+model.name+'/'+reranker.name
                        reranker.save_recommendations(saving_model_path)
            if self.dynamic_rerankers is not None:
                dyn_reranker_evaluator = DynamicReRankEval(self.eval_method)
                print(f"Dynamic reranking evaluation for model {model.name} started")
                for j in range(len(self.dynamic_rerankers)):
                    reranker = self.dynamic_rerankers[j]
                    reranker.reset()

                test_result_dyn_reranker, val_result_dyn_reranker = dyn_reranker_evaluator.evaluate(
                    model= model, metrics = self.metrics, user_based = self.user_based, rerankers = self.dynamic_rerankers, show_validation= self.show_validation)
                self.dynamic_reranking_result.append(test_result_dyn_reranker)
            
                if save_dir and (not isinstance(self.dynamic_reranking_result, CVExperimentResult)):
                    for reranker in self.dynamic_rerankers:
                        saving_model_path = save_dir + '/'+model.name+'/'+reranker.name
                        reranker.save_recommendations(saving_model_path)
            

        output = ""
        if self.val_result is not None:
            output += "\nVALIDATION:\n...\n{}".format(self.val_result)

        output += "\nTEST:\n...\n{}".format(self.result)
        if self.rerankers is not None:
            output += "\nStatic Re-Ranking:\n...\n{}".format(self.static_reranking_result)
        if self.dynamic_rerankers is not None:
            output += "\nDynamic Re-Ranking:\n...\n{}".format(self.dynamic_reranking_result)

        print(output)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

        # save_dir = "." if self.save_dir is None else self.save_dir
        output_file = os.path.join(save_dir, "CornacExp-{}.log".format(timestamp))
        with open(output_file, "w") as f:
            f.write(output)

        # model_results_dict = OrderedDict()
        # if not isinstance(self.result, CVExperimentResult) and self.result:
        #     for experiment_result in self.result:
        #         model_data = OrderedDict({
        #             "metric_avg_results": experiment_result.metric_avg_results,
        #             "user_info": experiment_result.user_info,
        #             "model_parameter": experiment_result.model_parameter,
        #             "static_rerank_avg_results": experiment_result.static_rerank_avg_results,
        #             "dynamic_rerank_avg_results": experiment_result.dynamic_rerank_avg_results
        #         })

                # Filter out empty values
                # model_data = {k: v for k, v in model_data.items() if v}

                # if model_data:
                #     model_results_dict[experiment_result.model_name] = model_data

        # Save results if there's any data
        # if model_results_dict:
        #     output_json_path = os.path.join(save_dir, f"CornacExp-{timestamp}.json")
        #     with open(output_json_path, "w") as f:
        #         json.dump(model_results_dict, f)

    

        
