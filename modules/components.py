import os
from typing import Any, Dict, Text

from absl import logging
from tfx import components
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2


def init_components(args: Dict[Text, Any]):
    try:
        output = example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ])
        )

        example_gen = components.CsvExampleGen(
            input_base=args["data_dir"],
            output_config=output
        )

        statistics_gen = components.StatisticsGen(
            examples=example_gen.outputs["examples"]
        )

        schema_gen = components.SchemaGen(
            statistics=statistics_gen.outputs["statistics"]
        )

        example_validator = components.ExampleValidator(
            statistics=statistics_gen.outputs["statistics"],
            schema=schema_gen.outputs["schema"],
        )

        transform = components.Transform(
            examples=example_gen.outputs["examples"],
            schema=schema_gen.outputs["schema"],
            module_file=os.path.abspath(args["transform_module"])
        )
        
        tuner = components.Tuner(
            module_file=os.path.abspath(args["tuner_module"]),
            examples=transform.outputs["transformed_examples"],
            transform_graph=transform.outputs["transform_graph"],
            schema=schema_gen.outputs["post_transform_schema"],
            train_args=trainer_pb2.TrainArgs(
                splits=["train"],
                num_steps=args["train_steps"],
            ),
            eval_args=trainer_pb2.EvalArgs(
                splits=["eval"],
                num_steps=args["eval_steps"],
            ),
            custom_config={"epochs": args["epochs"]}
        )

        trainer = components.Trainer(
            module_file=os.path.abspath(args["trainer_module"]),
            examples=transform.outputs["transformed_examples"],
            transform_graph=transform.outputs["transform_graph"],
            schema=transform.outputs["post_transform_schema"],
            hyperparameters=tuner.outputs["best_hyperparameters"],
            train_args=trainer_pb2.TrainArgs(
                splits=["train"],
                num_steps=args["train_steps"],
            ),
            eval_args=trainer_pb2.EvalArgs(
                splits=["eval"],
                num_steps=args["eval_steps"],
            ),
            custom_config={"epochs": args["epochs"]}
        )

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
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            tuner,
            trainer,
            pusher,
        )
    except BaseException as err:
        logging.error(f"ERROR IN init_components:\n{err}")