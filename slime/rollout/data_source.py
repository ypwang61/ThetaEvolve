import abc
import copy
import glob
import logging
import os
from pathlib import Path
import re
from typing import Optional

import torch

from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _find_latest_database_checkpoint(load_dir: str) -> Optional[int]:
    rollout_dir = os.path.join(load_dir, "rollout")
    if not os.path.exists(rollout_dir):
        return None

    db_files = glob.glob(os.path.join(rollout_dir, "evolving_gym_database_*"))
    if not db_files:
        return None

    rollout_ids = []
    for db_file in db_files:
        match = re.search(r"evolving_gym_database_(\d+)", os.path.basename(db_file))
        if match:
            rollout_ids.append(int(match.group(1)))
    return max(rollout_ids) if rollout_ids else None


class EvolvingGymManager:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        assert args.evolving_gym_initial_program and args.evolving_gym_evaluator_file, (
            "EvolvingGym needs --evolving-gym-initial-program and --evolving-gym-evaluator-file"
        )

        from openevolve.evolving_gym import SingleTaskEvolvingGym
        from slime.rollout.rm_hub.evolving_gym_rm import set_gym as set_evolving_gym_to_rm

        self.gym = SingleTaskEvolvingGym(
            initial_program_path=args.evolving_gym_initial_program,
            evaluation_file=args.evolving_gym_evaluator_file,
            config_path=getattr(args, "evolving_gym_config_path", None),
            config=None,
            max_concurrent_evaluations=getattr(args, "evolving_gym_max_concurrent_evals", 8),
            log_prompts=getattr(args, "evolving_gym_log_prompts", True),
            lazy_output_penalty_level=getattr(args, "evolving_gym_lazy_output_penalty_level", 2),
            database_reinit_ratio=getattr(args, "evolving_gym_database_reinit_ratio", 0.0),
            smallest_restart_step=getattr(args, "evolving_gym_smallest_restart_step", 0),
            largest_restart_step=getattr(args, "evolving_gym_largest_restart_step", None),
            add_historical_programs=getattr(args, "evolving_gym_add_historical_programs", 0),
            reward_process_type=args.evolving_gym_reward_process_type,
            seed=args.evolving_gym_seed,
        )

        if getattr(self.args, "evolving_gym_record", False):
            self.gym.enable_recording(getattr(self.args, "evolving_gym_record_dir", "gym_records"))

        # Make gym accessible to RM in this process.
        set_evolving_gym_to_rm(self.gym)

    def _ensure_initialized(self):
        if not getattr(self.gym, "_initialized", False):
            self.gym.initialize_sync()

    def get_sample(self) -> Sample:
        self._ensure_initialized()
        prompt_dict, parent_program = self.gym.problem_generator()
        system_txt = prompt_dict.get("system") or ""
        user_txt = prompt_dict.get("user") or ""

        if not self.args.apply_chat_template:
            raise RuntimeError("EvolvingGym requires --apply-chat-template in current integration.")

        messages = []
        if system_txt:
            messages.append({"role": "system", "content": system_txt})
        if not user_txt:
            raise RuntimeError("EvolvingGym prompt user message is empty.")
        messages.append({"role": "user", "content": user_txt})
        prompt_str = self.tokenizer.apply_chat_template(messages, None, tokenize=False, add_generation_prompt=True)

        return Sample(
            prompt=prompt_str,
            label=None,
            metadata={
                "parent_program": parent_program,
                "evolving_gym": True,
                "rm_type": "evolving-gym",
            },
        )


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to the data source
        """

    @abc.abstractmethod
    def save(self, rollout_id):
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """
        Load the state of the data source
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Length of the data source. May change when samples are added/fetched.
        """


# TODO may further refactor data-loading part later
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}
        self.evolving_gym_manager = None

        if args.rollout_global_dataset:
            tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")
                if processor:
                    processor.save_pretrained(Path(d) / "processor")

            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                processor=processor,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                multimodal_keys=args.multimodal_keys,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        elif getattr(args, "evolving_gym", False):
            self.dataset = None
            tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            self.evolving_gym_manager = EvolvingGymManager(args, tokenizer)
        else:
            self.dataset = None

    def get_samples(self, num_samples):
        # TODO further improve code
        if self.dataset is not None:
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
            else:
                prompt_samples = self.dataset.samples[self.sample_offset :]
                num_samples -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:num_samples]
                self.sample_offset = num_samples
        elif self.evolving_gym_manager is not None:
            prompt_samples = []
            while len(prompt_samples) < num_samples:
                prompt_samples.append(self.evolving_gym_manager.get_sample())
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if getattr(self.args, "evolving_gym", False):
            database_path = os.path.join(self.args.save, f"rollout/evolving_gym_database_{rollout_id}")
            os.makedirs(os.path.dirname(database_path), exist_ok=True)
            self.evolving_gym_manager.gym.database.save(database_path, rollout_id)
        elif not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if self.args.load is None:
            if getattr(self.args, "evolving_gym", False) and self.evolving_gym_manager is not None:
                self.evolving_gym_manager._ensure_initialized()
            return None

        detected_rollout_id = None
        if rollout_id == -1 and getattr(self.args, "evolving_gym", False):
            detected_rollout_id = _find_latest_database_checkpoint(self.args.load)
            if detected_rollout_id is None:
                logger.info("No evolving-gym database checkpoint found, initializing database.")
                self.evolving_gym_manager._ensure_initialized()
                return None
            rollout_id = detected_rollout_id

        if getattr(self.args, "evolving_gym", False):
            database_path = os.path.join(self.args.load, f"rollout/evolving_gym_database_{rollout_id}")
            if os.path.exists(database_path):
                self.evolving_gym_manager.gym.database.load(database_path)
                self.evolving_gym_manager.gym._initialized = True
            else:
                logger.info(f"Evolving gym database {database_path} does not exist, initializing database.")
                self.evolving_gym_manager._ensure_initialized()
                return detected_rollout_id

        if not self.args.rollout_global_dataset:
            return detected_rollout_id

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return detected_rollout_id

        logger.info(f"load metadata from {path}")
        logger.info(f"load metadata: {self.metadata}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        if self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)

        return detected_rollout_id

    def __len__(self) -> int:
        return len(self.dataset) if self.dataset is not None else 0


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
