import os
from typing import Any, Dict, Text

from absl import logging
from tfx import components
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import \
    LatestBlessedModelStrategy
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing


def init_components(args: Dict[Text, Any]):
    try:
        output = example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(
                    name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ])
        )

        movies_example_gen = components.CsvExampleGen(
            input_base=args["movies_data_dir"],
            output_config=output
        )

        rating_example_gen = components.CsvExampleGen(
            input_base=args["rating_data_dir"],
            output_config=output
        )

        movies_statistics_gen = components.StatisticsGen(
            examples=movies_example_gen.outputs["examples"]
        )

        rating_statistics_gen = components.StatisticsGen(
            examples=rating_example_gen.outputs["examples"]
        )

        movies_schema_gen = components.SchemaGen(
            statistics=movies_statistics_gen.outputs["statistics"]
        )

        rating_schema_gen = components.SchemaGen(
            statistics=rating_statistics_gen.outputs["statistics"]
        )

        movies_example_validator = components.ExampleValidator(
            statistics=movies_statistics_gen.outputs["statistics"],
            schema=movies_schema_gen.outputs["schema"],
        )

        rating_example_validator = components.ExampleValidator(
            statistics=rating_statistics_gen.outputs["statistics"],
            schema=rating_schema_gen.outputs["schema"],
        )

        movies_transform = components.Transform(
            examples=movies_example_gen.outputs["examples"],
            schema=movies_schema_gen.outputs["schema"],
            module_file=os.path.abspath(args["movies_transform_module"])
        )

        rating_transform = components.Transform(
            examples=rating_example_gen.outputs["examples"],
            schema=rating_schema_gen.outputs["schema"],
            module_file=os.path.abspath(args["rating_transform_module"])
        )

        # tuner = components.Tuner(
        #     module_file=os.path.abspath(args["tuner_module"]),
        #     examples=transform.outputs["transformed_examples"],
        #     transform_graph=transform.outputs["transform_graph"],
        #     schema=transform.outputs["post_transform_schema"],
        #     train_args=trainer_pb2.TrainArgs(
        #         splits=["train"],
        #         num_steps=args["train_steps"],
        #     ),
        #     eval_args=trainer_pb2.EvalArgs(
        #         splits=["eval"],
        #         num_steps=args["eval_steps"],
        #     ),
        #     custom_config={
        #         "epochs": args["epochs"],
        #     }
        # )

        trainer = components.Trainer(
            module_file=os.path.abspath(args["trainer_module"]),
            examples=rating_transform.outputs["transformed_examples"],
            transform_graph=rating_transform.outputs["transform_graph"],
            schema=rating_transform.outputs["post_transform_schema"],
            # hyperparameters=tuner.outputs["best_hyperparameters"],
            train_args=trainer_pb2.TrainArgs(
                splits=["train"],
                num_steps=args["train_steps"],
            ),
            eval_args=trainer_pb2.EvalArgs(
                splits=["eval"],
                num_steps=args["eval_steps"],
            ),
            custom_config={
                "epochs": args["epochs"],
                "movies": movies_transform.outputs["transformed_examples"],
                "movies_schema": movies_transform.outputs["post_transform_schema"],
            }
        )

        model_resolver = Resolver(
            strategy_class=LatestBlessedModelStrategy,
            model=Channel(type=Model),
            model_blessing=Channel(type=ModelBlessing),
        ).with_id("Latest_blessed_model_resolve")

        pusher = components.Pusher(
            model=trainer.outputs["model"],
            push_destination=pusher_pb2.PushDestination(
                filesystem=pusher_pb2.PushDestination.Filesystem(
                    base_directory=os.path.join(
                        args["serving_model_dir"], "movie-recommender"
                    ),
                )
            )
        )

        return (
            movies_example_gen,
            rating_example_gen,
            movies_statistics_gen,
            rating_statistics_gen,
            movies_schema_gen,
            rating_schema_gen,
            movies_example_validator,
            rating_example_validator,
            movies_transform,
            rating_transform,
            # tuner,
            trainer,
            model_resolver,
            pusher,
        )
    except BaseException as err:
        logging.error(f"ERROR IN init_components:\n{err}")
