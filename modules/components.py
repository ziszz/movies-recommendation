import os

import tensorflow_model_analysis as tfma
from absl import logging
from tfx import components
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import \
    LatestBlessedModelStrategy
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

from modules.transform import FEATURE_KEYS, LABEL_KEY, transformed_name


def init_components(**kwargs):
    try:
        output = example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(
                    name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(
                    name="eval", hash_buckets=2),
            ])
        )

        example_gen = components.CsvExampleGen(
            input_base=kwargs["data_dir"],
            output_config=output,
        )

        statistics_gen = components.StatisticsGen(
            examples=example_gen.outputs["examples"],
        )

        schema_gen = components.SchemaGen(
            statistics=statistics_gen.outputs["statistics"],
        )

        example_validator = components.ExampleValidator(
            statistics=statistics_gen.outputs["statistics"],
            schema=schema_gen.outputs["schema"],
        )

        transform = components.Transform(
            examples=example_gen.outputs["examples"],
            schema=schema_gen.outputs["schema"],
            module_file=os.path.abspath(kwargs["transform_module"]),
        )

        tuner = components.Tuner(
            module_file=os.path.abspath(kwargs["tuner_module"]),
            examples=transform.outputs["transformed_examples"],
            transform_graph=transform.outputs["transform_graph"],
            schema=transform.outputs["post_transform_schema"],
            train_args=trainer_pb2.TrainArgs(
                splits=["train"],
                num_steps=kwargs["train_steps"],
            ),
            eval_args=trainer_pb2.EvalArgs(
                splits=["eval"],
                num_steps=kwargs["eval_steps"],
            ),
            custom_config={
                "epochs": kwargs["epochs"],
            }
        )

        trainer = components.Trainer(
            module_file=os.path.abspath(kwargs["trainer_module"]),
            examples=transform.outputs["transformed_examples"],
            transform_graph=transform.outputs["transform_graph"],
            schema=transform.outputs["post_transform_schema"],
            hyperparameters=tuner.outputs["best_hyperparameters"],
            train_args=trainer_pb2.TrainArgs(
                splits=["train"],
                num_steps=kwargs["train_steps"],
            ),
            eval_args=trainer_pb2.EvalArgs(
                splits=["eval"],
                num_steps=kwargs["eval_steps"],
            ),
        )

        model_resolver = Resolver(
            strategy_class=LatestBlessedModelStrategy,
            model=Channel(type=Model),
            model_blessing=Channel(type=ModelBlessing),
        ).with_id("Latest_blessed_model_resolve")

        eval_config = tfma.EvalConfig(
            model_specs=[tfma.ModelSpec(
                label_key=transformed_name(LABEL_KEY))],
            slicing_specs=[
                tfma.SlicingSpec(feature_keys=[
                    transformed_name(f) for f in FEATURE_KEYS
                ]),
            ],
            metrics_specs=[
                tfma.MetricsSpec(metrics=[
                    tfma.MetricConfig(
                        class_name="MeanSquaredError",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.1},
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.LOWER_IS_BETTER,
                                absolute={"value": 1.0},
                            ),
                        ),
                    ),
                ])
            ]
        )

        evaluator = components.Evaluator(
            examples=example_gen.outputs["examples"],
            model=trainer.outputs["model"],
            baseline_model=model_resolver.outputs["model"],
            eval_config=eval_config,
        )

        pusher = components.Pusher(
            model=trainer.outputs["model"],
            push_destination=pusher_pb2.PushDestination(
                filesystem=pusher_pb2.PushDestination.Filesystem(
                    base_directory=os.path.join(
                        kwargs["serving_model_dir"],
                        "cf-model",
                    ),
                )
            )
        )

        return (
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            tuner,
            trainer,
            model_resolver,
            evaluator,
            pusher,
        )
    except BaseException as err:
        logging.error(f"ERROR IN init_components:\n{err}")
