import csv
import json
import logging
import os
import pickle
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from os.path import abspath, exists
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from transformers import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, AutoConfig
from transformers import BartConfig
from transformers import DistilBertConfig
from transformers import RobertaConfig
from transformers import T5Config
from transformers import PretrainedConfig
from transformers import XLMConfig
from .data import SquadExample, squad_convert_examples_to_features
from transformers import is_tf_available, is_torch_available
from transformers import ModelCard
from transformers import AutoTokenizer
from transformers import BasicTokenizer
from transformers import PreTrainedTokenizer
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
)


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from input_reader import Namespace


if is_tf_available():
    import tensorflow as tf
    from transformers import (
        TFAutoModel,
        TFAutoModelForSequenceClassification,
        TFAutoModelForQuestionAnswering,
        TFAutoModelForTokenClassification,
        TFAutoModelWithLMHead,
    )

if is_torch_available():
    import torch
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoModelForQuestionAnswering,
        AutoModelForTokenClassification,
        AutoModelWithLMHead,
    )

from .utils_squad import (
    SquadExample,
    convert_examples_to_features,
    RawResult,
    RawResultExtended,
    get_predictions_xlnet,
)

logger = logging.getLogger(__name__)


class CompatSquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        doc_tokens,
        orig_answer_text=None,
        start_position=None,
        end_position=None,
        is_impossible=None,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


def get_framework(model=None):
    """ Select framework (TensorFlow/PyTorch) to use.
        If both frameworks are installed and no specific model is provided, defaults to using PyTorch.
    """
    if (
        is_tf_available()
        and is_torch_available()
        and model is not None
        and not isinstance(model, str)
    ):
        # Both framework are available but the user supplied a model class instance.
        # Try to guess which framework to use from the model classname
        framework = "tf" if model.__class__.__name__.startswith("TF") else "pt"
    elif not is_tf_available() and not is_torch_available():
        raise RuntimeError(
            "At least one of TensorFlow 2.0 or PyTorch should be installed. "
            "To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ "
            "To install PyTorch, read the instructions at https://pytorch.org/."
        )
    else:
        # framework = 'tf' if is_tf_available() else 'pt'
        framework = "pt" if is_torch_available() else "tf"
    return framework


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


class ArgumentHandler(ABC):
    """
    Base interface for handling varargs for each Pipeline
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class DefaultArgumentHandler(ArgumentHandler):
    """
    Default varargs argument parser handling parameters for each Pipeline
    """

    def __call__(self, *args, **kwargs):
        if "X" in kwargs:
            return kwargs["X"]
        elif "data" in kwargs:
            return kwargs["data"]
        elif len(args) == 1:
            if isinstance(args[0], list):
                return args[0]
            else:
                return [args[0]]
        elif len(args) > 1:
            return list(args)
        raise ValueError(
            "Unable to infer the format of the provided data (X=, data=, ...)"
        )


class PipelineDataFormat:
    """
    Base class for all the pipeline supported data format both for reading and writing.
    Supported data formats currently includes:
     - JSON
     - CSV
     - stdin/stdout (pipe)
    PipelineDataFormat also includes some utilities to work with multi-columns like mapping from datasets columns
    to pipelines keyword arguments through the `dataset_kwarg_1=dataset_column_1` format.
    """

    SUPPORTED_FORMATS = ["json", "csv", "pipe"]

    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ):
        self.output_path = output_path
        self.input_path = input_path
        self.column = column.split(",") if column is not None else [""]
        self.is_multi_columns = len(self.column) > 1

        if self.is_multi_columns:
            self.column = [
                tuple(c.split("=")) if "=" in c else (c, c) for c in self.column
            ]

        if output_path is not None and not overwrite:
            if exists(abspath(self.output_path)):
                raise OSError("{} already exists on disk".format(self.output_path))

        if input_path is not None:
            if not exists(abspath(self.input_path)):
                raise OSError("{} doesnt exist on disk".format(self.input_path))

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self, data: dict):
        """
        Save the provided data object with the representation for the current `DataFormat`.
        :param data: data to store
        :return:
        """
        raise NotImplementedError()

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        """
        Save the provided data object as a pickle-formatted binary data on the disk.
        :param data: data to store
        :return: (str) Path where the data has been saved
        """
        path, _ = os.path.splitext(self.output_path)
        binary_path = os.path.extsep.join((path, "pickle"))

        with open(binary_path, "wb+") as f_output:
            pickle.dump(data, f_output)

        return binary_path

    @staticmethod
    def from_str(
        format: str,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ):
        if format == "json":
            return JsonPipelineDataFormat(
                output_path, input_path, column, overwrite=overwrite
            )
        elif format == "csv":
            return CsvPipelineDataFormat(
                output_path, input_path, column, overwrite=overwrite
            )
        elif format == "pipe":
            return PipedPipelineDataFormat(
                output_path, input_path, column, overwrite=overwrite
            )
        else:
            raise KeyError(
                "Unknown reader {} (Available reader are json/csv/pipe)".format(format)
            )


class CsvPipelineDataFormat(PipelineDataFormat):
    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ):
        super().__init__(output_path, input_path, column, overwrite=overwrite)

    def __iter__(self):
        with open(self.input_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.is_multi_columns:
                    yield {k: row[c] for k, c in self.column}
                else:
                    yield row[self.column[0]]

    def save(self, data: List[dict]):
        with open(self.output_path, "w") as f:
            if len(data) > 0:
                writer = csv.DictWriter(f, list(data[0].keys()))
                writer.writeheader()
                writer.writerows(data)


class JsonPipelineDataFormat(PipelineDataFormat):
    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ):
        super().__init__(output_path, input_path, column, overwrite=overwrite)

        with open(input_path, "r") as f:
            self._entries = json.load(f)

    def __iter__(self):
        for entry in self._entries:
            if self.is_multi_columns:
                yield {k: entry[c] for k, c in self.column}
            else:
                yield entry[self.column[0]]

    def save(self, data: dict):
        with open(self.output_path, "w") as f:
            json.dump(data, f)


class PipedPipelineDataFormat(PipelineDataFormat):
    """
    Read data from piped input to the python process.
    For multi columns data, columns should separated by \t
    If columns are provided, then the output will be a dictionary with {column_x: value_x}
    """

    def __iter__(self):
        for line in sys.stdin:
            # Split for multi-columns
            if "\t" in line:

                line = line.split("\t")
                if self.column:
                    # Dictionary to map arguments
                    yield {kwargs: l for (kwargs, _), l in zip(self.column, line)}
                else:
                    yield tuple(line)

            # No dictionary to map arguments
            else:
                yield line

    def save(self, data: dict):
        print(data)

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        if self.output_path is None:
            raise KeyError(
                "When using piped input on pipeline outputting large object requires an output file path. "
                "Please provide such output path through --output argument."
            )

        return super().save_binary(data)


class _ScikitCompat(ABC):
    """
    Interface layer for the Scikit and Keras compatibility.
    """

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()


class Pipeline(_ScikitCompat):
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.
    Base class implementing pipelined operations.
    Pipeline workflow is defined as a sequence of the following operations:
        Input -> Tokenization -> Model Inference -> Post-Processing (Task dependent) -> Output
    Pipeline supports running on CPU or GPU through the device argument. Users can specify
    device argument as an integer, -1 meaning "CPU", >= 0 referring the CUDA device ordinal.
    Some pipeline, like for instance FeatureExtractionPipeline ('feature-extraction') outputs large
    tensor object as nested-lists. In order to avoid dumping such large structure as textual data we
    provide the binary_output constructor argument. If set to True, the output will be stored in the
    pickle format.
    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.
            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
        binary_output (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Flag indicating if the output the pipeline should happen in a binary format (i.e. pickle) or as raw text.
    Return:
        :obj:`List` or :obj:`Dict`:
        Pipeline returns list or dictionary depending on:
         - Whether the user supplied multiple samples
         - Whether the pipeline exposes multiple fields in the output object
    """

    default_input_names = None

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        task: str = "",
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
    ):

        if framework is None:
            framework = get_framework()

        self.model = model
        self.tokenizer = tokenizer
        self.modelcard = modelcard
        self.framework = framework
        self.device = (
            device
            if framework == "tf"
            else torch.device("cpu" if device < 0 else "cuda:{}".format(device))
        )
        self.binary_output = binary_output
        self._args_parser = args_parser or DefaultArgumentHandler()

        # Special handling
        if self.framework == "pt" and self.device.type == "cuda":
            self.model = self.model.to(self.device)

        # Update config with task specific parameters
        task_specific_params = self.model.config.task_specific_params
        if task_specific_params is not None and task in task_specific_params:
            self.model.config.update(task_specific_params.get(task))

    def save_pretrained(self, save_directory):
        """
        Save the pipeline's model and tokenizer to the specified save_directory
        """
        if not os.path.isdir(save_directory):
            logger.error(
                "Provided path ({}) should be a directory".format(save_directory)
            )
            return

        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        if self.modelcard is not None:
            self.modelcard.save_pretrained(save_directory)

    def transform(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    def predict(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    @contextmanager
    def device_placement(self):
        """
        Context Manager allowing tensor allocation on the user-specified device in framework agnostic way.
        example:
            # Explicitly ask for tensor allocation on CUDA device :0
            nlp = pipeline(..., device=0)
            with nlp.device_placement():
                # Every framework specific tensor allocation will be done on the request device
                output = nlp(...)
        Returns:
            Context manager
        """
        if self.framework == "tf":
            with tf.device(
                "/CPU:0" if self.device == -1 else "/device:GPU:{}".format(self.device)
            ):
                yield
        else:
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)

            yield

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.
        :param inputs:
        :return:
        """
        return {name: tensor.to(self.device) for name, tensor in inputs.items()}

    def inputs_for_model(self, features: Union[dict, List[dict]]) -> Dict:
        """
        Generates the input dictionary with model-specific parameters.
        Returns:
            dict holding all the required parameters for model's forward
        """
        args = ["input_ids", "attention_mask"]

        if not isinstance(
            self.model.config,
            (DistilBertConfig, XLMConfig, RobertaConfig, BartConfig, T5Config),
        ):
            args += ["token_type_ids"]

        # PR #1548 (CLI) There is an issue with attention_mask
        # if 'xlnet' in model_type or 'xlm' in model_type:
        #     args += ['cls_index', 'p_mask']

        if isinstance(features, dict):
            return {k: features[k] for k in args}
        else:
            return {k: [feature[k] for feature in features] for k in args}

    def _parse_and_tokenize(self, *texts, pad_to_max_length=False, **kwargs):
        """
        Parse arguments and tokenize
        """
        # Parse arguments
        inputs = self._args_parser(*texts, **kwargs)
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            add_special_tokens=True,
            return_tensors=self.framework,
            max_length=self.tokenizer.max_len,
            pad_to_max_length=pad_to_max_length,
        )

        # Filter out features not available on specific models
        inputs = self.inputs_for_model(inputs)

        return inputs

    def __call__(self, *texts, **kwargs):
        inputs = self._parse_and_tokenize(*texts, **kwargs)
        return self._forward(inputs)

    def _forward(self, inputs, return_tensors=False):
        """
        Internal framework specific forward dispatching.
        Args:
            inputs: dict holding all the keyworded arguments for required by the model forward method.
            return_tensors: Whether to return native framework (pt/tf) tensors rather than numpy array.
        Returns:
            Numpy array
        """
        # Encode for forward
        with self.device_placement():
            if self.framework == "tf":
                # TODO trace model
                predictions = self.model(inputs, training=False)[0]
            else:
                with torch.no_grad():
                    inputs = self.ensure_tensor_on_device(**inputs)
                    predictions = self.model(**inputs)[0].cpu()

        if return_tensors:
            return predictions
        else:
            return predictions.numpy()


class FeatureExtractionPipeline(Pipeline):
    """
    Feature extraction pipeline using Model head. This pipeline extracts the hidden states from the base transformer,
    which can be used as features in a downstream tasks.
    This feature extraction pipeline can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):
    - "feature-extraction", for extracting features of a sequence.
    All models may be used for this pipeline. See a list of all models, including community-contributed models on
    `huggingface.co/models <https://huggingface.co/models>`__.
    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.
            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        task: str = "",
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=True,
            task=task,
        )

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs).tolist()


class TextClassificationPipeline(Pipeline):
    """
    Text classification pipeline using ModelForSequenceClassification head. See the
    `sequence classification usage <../usage.html#sequence-classification>`__ examples for more information.
    This text classification pipeline can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):
    - "sentiment-analysis", for classifying sequences according to positive or negative sentiments.
    The models that this pipeline can use are models that have been fine-tuned on a sequence classification task.
    See the list of available community models fine-tuned on such a task on
    `huggingface.co/models <https://huggingface.co/models?search=&filter=text-classification>`__.
    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.
            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    def __call__(self, *args, **kwargs):
        outputs = super().__call__(*args, **kwargs)
        scores = np.exp(outputs) / np.exp(outputs).sum(-1)
        return [
            {"label": self.model.config.id2label[item.argmax()], "score": item.max()}
            for item in scores
        ]


class FillMaskPipeline(Pipeline):
    """
    Masked language modeling prediction pipeline using ModelWithLMHead head. See the
    `masked language modeling usage <../usage.html#masked-language-modeling>`__ examples for more information.
    This mask filling pipeline can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):
    - "fill-mask", for predicting masked tokens in a sequence.
    The models that this pipeline can use are models that have been trained with a masked language modeling objective,
    which includes the bi-directional models in the library.
    See the list of available community models on
    `huggingface.co/models <https://huggingface.co/models?search=&filter=lm-head>`__.
    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.
            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        topk=5,
        task: str = "",
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=True,
            task=task,
        )

        self.topk = topk

    def __call__(self, *args, **kwargs):
        inputs = self._parse_and_tokenize(*args, **kwargs)
        outputs = self._forward(inputs, return_tensors=True)

        results = []
        batch_size = outputs.shape[0] if self.framework == "tf" else outputs.size(0)

        for i in range(batch_size):
            input_ids = inputs["input_ids"][i]
            result = []

            if self.framework == "tf":
                masked_index = (
                    tf.where(input_ids == self.tokenizer.mask_token_id).numpy().item()
                )
                logits = outputs[i, masked_index, :]
                probs = tf.nn.softmax(logits)
                topk = tf.math.top_k(probs, k=self.topk)
                values, predictions = topk.values.numpy(), topk.indices.numpy()
            else:
                masked_index = (
                    (input_ids == self.tokenizer.mask_token_id).nonzero().item()
                )
                logits = outputs[i, masked_index, :]
                probs = logits.softmax(dim=0)
                values, predictions = probs.topk(self.topk)

            for v, p in zip(values.tolist(), predictions.tolist()):
                tokens = input_ids.numpy()
                tokens[masked_index] = p
                # Filter padding out:
                tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
                result.append(
                    {"sequence": self.tokenizer.decode(tokens), "score": v, "token": p}
                )

            # Append
            results += [result]

        if len(results) == 1:
            return results[0]
        return results


class NerPipeline(Pipeline):
    """
    Named Entity Recognition pipeline using ModelForTokenClassification head. See the
    `named entity recognition usage <../usage.html#named-entity-recognition>`__ examples for more information.
    This token recognition pipeline can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):
    - "ner", for predicting the classes of tokens in a sequence: person, organisation, location or miscellaneous.
    The models that this pipeline can use are models that have been fine-tuned on a token classification task.
    See the list of available community models fine-tuned on such a task on
    `huggingface.co/models <https://huggingface.co/models?search=&filter=token-classification>`__.
    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.
            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    default_input_names = "sequences"

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
        ignore_labels=["O"],
        task: str = "",
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=binary_output,
            task=task,
        )

        self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self.ignore_labels = ignore_labels

    def __call__(self, *texts, **kwargs):
        inputs = self._args_parser(*texts, **kwargs)
        answers = []
        for sentence in inputs:

            # Manage correct placement of the tensors
            with self.device_placement():

                tokens = self.tokenizer.encode_plus(
                    sentence,
                    return_attention_mask=False,
                    return_tensors=self.framework,
                    max_length=self.tokenizer.max_len,
                )

                # Forward
                if self.framework == "tf":
                    entities = self.model(tokens)[0][0].numpy()
                    input_ids = tokens["input_ids"].numpy()[0]
                else:
                    with torch.no_grad():
                        tokens = self.ensure_tensor_on_device(**tokens)
                        entities = self.model(**tokens)[0][0].cpu().numpy()
                        input_ids = tokens["input_ids"].cpu().numpy()[0]

            score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
            labels_idx = score.argmax(axis=-1)

            answer = []
            for idx, label_idx in enumerate(labels_idx):
                if self.model.config.id2label[label_idx] not in self.ignore_labels:
                    answer += [
                        {
                            "word": self.tokenizer.convert_ids_to_tokens(
                                int(input_ids[idx])
                            ),
                            "score": score[idx][label_idx].item(),
                            "entity": self.model.config.id2label[label_idx],
                        }
                    ]

            # Append
            answers += [answer]
        if len(answers) == 1:
            return answers[0]
        return answers


TokenClassificationPipeline = NerPipeline


class QuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    QuestionAnsweringPipeline requires the user to provide multiple arguments (i.e. question & context) to be mapped
    to internal SquadExample / SquadFeature structures.
    QuestionAnsweringArgumentHandler manages all the possible to create SquadExample from the command-line supplied
    arguments.
    """

    def __call__(self, *args, **kwargs):

        inputs = []

        if "question" in kwargs and "context" in kwargs:
            question = kwargs["question"]
            context = kwargs["context"]
            if isinstance(kwargs["question"], str) and isinstance(
                kwargs["context"], str
            ):
                paragraphs["qas"] = [
                    {"question": question, "id": question, "is_impossible": False,}
                ]
                paragraphs["context"] = context
                inputs.append({"paragraphs": [paragraphs]})
            elif isinstance(kwargs["question"], str) and isinstance(
                kwargs["context"], list
            ):

                paragraphs = {}
                paragraphs["qas"] = [
                    {"question": question, "id": question, "is_impossible": False,}
                ]
                paragraphs["context"] = "\n\n".join(context)
                inputs.append({"paragraphs": [paragraphs]})

        # elif isinstance(kwargs["question"], str) and isinstance(
        #     kwargs["context"], list
        # ):
        #     i = 1
        #     for cont in context:
        #         paragraphs = {}
        #         paragraphs["qas"] = [
        #             {
        #                 "question": question,
        #                 "id": question + "_" + str(i),
        #                 "is_impossible": False,
        #             }
        #         ]
        #         paragraphs["context"] = cont
        #         inputs.append({"paragraphs": [paragraphs]})
        #         i += 1

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        examples = []
        for entry in inputs:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False

                    example = CompatSquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible,
                    )
                    examples.append(example)
        if (
            "question" in kwargs
            and "context" in kwargs
            and isinstance(kwargs["question"], str)
            and isinstance(kwargs["context"], list)
        ):
            return examples, kwargs["context"], kwargs["question"]

        return examples, None, None


class QuestionAnsweringPipeline(Pipeline):
    """
    Question Answering pipeline using ModelForQuestionAnswering head. See the
    `question answering usage <../usage.html#question-answering>`__ examples for more information.
    This question answering can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):
    - "question-answering", for answering questions given a context.
    The models that this pipeline can use are models that have been fine-tuned on a question answering task.
    See the list of available community models fine-tuned on such a task on
    `huggingface.co/models <https://huggingface.co/models?search=&filter=question-answering>`__.
    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.
            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    default_input_names = "question,context"

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        device: int = -1,
        task: str = "",
        **kwargs
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=QuestionAnsweringArgumentHandler(),
            device=device,
            task=task,
            **kwargs,
        )

    @staticmethod
    def create_sample(
        question: Union[str, List[str]], context: Union[str, List[str]]
    ) -> Union[SquadExample, List[SquadExample]]:
        """
        QuestionAnsweringPipeline leverages the SquadExample/SquadFeatures internally.
        This helper method encapsulate all the logic for converting question(s) and context(s) to SquadExample(s).
        We currently support extractive question answering.
        Arguments:
             question: (str, List[str]) The question to be ask for the associated context
             context: (str, List[str]) The context in which we will look for the answer.
        Returns:
            SquadExample initialized with the corresponding question and context.
        """
        if isinstance(question, list):
            return [
                SquadExample(None, q, c, None, None, None)
                for q, c in zip(question, context)
            ]
        else:
            return SquadExample(None, question, context, None, None, None)

    def __call__(self, *texts, **kwargs):
        """
        Args:
            We support multiple use-cases, the following are exclusive:
            X: sequence of SquadExample
            data: sequence of SquadExample
            question: (str, List[str]), batch of question(s) to map along with context
            context: (str, List[str]), batch of context(s) associated with the provided question keyword argument
        Returns:
            dict: {'answer': str, 'score": float, 'start": int, "end": int}
            answer: the textual answer in the intial context
            score: the score the current answer scored for the model
            start: the character index in the original string corresponding to the beginning of the answer' span
            end: the character index in the original string corresponding to the ending of the answer' span
        """
        # Set defaults values
        kwargs.setdefault("topk", 1)
        kwargs.setdefault("doc_stride", 128)
        kwargs.setdefault("max_answer_len", 15)
        kwargs.setdefault("max_seq_len", 2000)
        kwargs.setdefault("max_question_len", 64)
        args = Namespace(
            model_type="xlnet",
            version_2_with_negative=True,
            max_seq_length=2000,
            doc_stride=128,
            max_query_length=64,
            do_eval=True,
            per_gpu_eval_batch_size=8,
            n_best_size=20,
            max_answer_length=30,
            no_cuda=False,
            seed=42,
            local_rank=-1,
            fp16_opt_level="01",
            server_ip="",
            server_port="",
            verbose_logging=False,
        )

        # Setup CUDA, GPU & distributed training
        if (
            vars(args)["_defaults"]["local_rank"] == -1
            or vars(args)["_defaults"]["no_cuda"]
        ):
            device = torch.device(
                "cuda"
                if torch.cuda.is_available() and not vars(args)["_defaults"]["no_cuda"]
                else "cpu"
            )
            args.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(vars(args)["_defaults"]["local_rank"])
            device = torch.device("cuda", vars(args)["_defaults"]["local_rank"])
            torch.distributed.init_process_group(backend="nccl")
            args.n_gpu = 1
        args.device = device
        args.model_type = "xlnet"
        args.do_eval = True

        # Select device (distibuted not available)
        args.finalize()

        if kwargs["topk"] < 1:
            raise ValueError(
                "topk parameter should be >= 1 (got {})".format(kwargs["topk"])
            )

        if kwargs["max_answer_len"] < 1:
            raise ValueError(
                "max_answer_len parameter should be >= 1 (got {})".format(
                    kwargs["max_answer_len"]
                )
            )
        # Convert inputs to features
        examples, contexts, question = self._args_parser(*texts, **kwargs)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=kwargs["max_seq_len"],
            doc_stride=kwargs["doc_stride"],
            max_query_length=kwargs["max_question_len"],
            is_training=False,
        )
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_example_index,
            all_cls_index,
            all_p_mask,
        )

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        sampler = (
            SequentialSampler(dataset)
            if args.local_rank == -1
            else DistributedSampler(dataset)
        )
        dataloader = DataLoader(
            dataset, sampler=sampler, batch_size=args.eval_batch_size
        )
        self.model.to(args.device)

        all_results = []
        for batch in tqdm(dataloader, desc="Answering"):
            self.model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                example_indices = batch[3]
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                outputs = self.model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(
                    unique_id=unique_id,
                    start_top_log_probs=to_list(outputs[0][i]),
                    start_top_index=to_list(outputs[1][i]),
                    end_top_log_probs=to_list(outputs[2][i]),
                    end_top_index=to_list(outputs[3][i]),
                    cls_logits=to_list(outputs[4][i]),
                )
                all_results.append(result)

        # XLNet uses a more complex post-processing procedure
        all_answers = get_predictions_xlnet(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            self.model.config.start_n_top,
            self.model.config.end_n_top,
            args.version_2_with_negative,
            self.tokenizer,
            args.verbose_logging,
        )
        result = []
        if contexts:
            answers = json.loads(all_answers)[question]
            for answer in answers:
                for context in contexts:
                    if answer["answer"] in context:
                        answer["context"] = contexts.index(context)
                        result.append(answer)
        return json.dumps(result, indent=4)


class SummarizationPipeline(Pipeline):
    """
    Summarize news articles and other documents
    Usage::
        # use bart in pytorch
        summarizer = pipeline("summarization")
        summarizer("Sam Shleifer writes the best docstring examples in the whole world.", min_length=5, max_length=20)
        # use t5 in tf
        summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
        summarizer("Sam Shleifer writes the best docstring examples in the whole world.", min_length=5, max_length=20)
    Supported Models:
        The models that this pipeline can use are models that have been fine-tuned on a summarization task, which is currently, '`bart-large-cnn`', '`t5-small`', '`t5-base`', '`t5-large`', '`t5-3b`', '`t5-11b`'.
    Arguments:
        model (:obj:`str` or :obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`, `optional`, defaults to :obj:`None`):
            The model that will be used by the pipeline to make predictions. This can be :obj:`None`, a string
            checkpoint identifier or an actual pre-trained model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
            If :obj:`None`, the default of the pipeline will be loaded.
        tokenizer (:obj:`str` or :obj:`~transformers.PreTrainedTokenizer`, `optional`, defaults to :obj:`None`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be :obj:`None`,
            a string checkpoint identifier or an actual pre-trained tokenizer inheriting from
            :class:`~transformers.PreTrainedTokenizer`.
            If :obj:`None`, the default of the pipeline will be loaded.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.
            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    def __call__(
        self,
        *documents,
        return_tensors=False,
        return_text=True,
        clean_up_tokenization_spaces=False,
        **generate_kwargs
    ):
        r"""
        Args:
            *documents: (list of strings) articles to be summarized
            return_text: (bool, default=True) whether to add a decoded "summary_text" to each result
            return_tensors: (bool, default=False) whether to return the raw "summary_token_ids" to each result
            clean_up_tokenization_spaces: (`optional`) bool whether to include extra spaces in the output
            **generate_kwargs: extra kwargs passed to `self.model.generate`_
        Returns:
            list of dicts with 'summary_text' and/or 'summary_token_ids' for each document_to_summarize
        .. _`self.model.generate`:
            https://huggingface.co/transformers/model_doc/bart.html#transformers.BartForConditionalGeneration.generate
        """
        assert (
            return_tensors or return_text
        ), "You must specify return_tensors=True or return_text=True"
        assert len(documents) > 0, "Please provide a document to summarize"

        if (
            self.framework == "tf"
            and "BartForConditionalGeneration" in self.model.__class__.__name__
        ):
            raise NotImplementedError(
                "Tensorflow is not yet supported for Bart. Please consider using T5, e.g. `t5-base`"
            )

        prefix = (
            self.model.config.prefix if self.model.config.prefix is not None else ""
        )

        if isinstance(documents[0], list):
            assert (
                self.tokenizer.pad_token_id is not None
            ), "Please make sure that the tokenizer has a pad_token_id when using a batch input"

            documents = ([prefix + document for document in documents[0]],)
            pad_to_max_length = True

        elif isinstance(documents[0], str):
            documents = (prefix + documents[0],)
            pad_to_max_length = False
        else:
            raise ValueError(
                " `documents[0]`: {} have the wrong format. The should be either of type `str` or type `list`".format(
                    documents[0]
                )
            )

        with self.device_placement():
            inputs = self._parse_and_tokenize(
                *documents, pad_to_max_length=pad_to_max_length
            )

            if self.framework == "pt":
                inputs = self.ensure_tensor_on_device(**inputs)
                input_length = inputs["input_ids"].shape[-1]
            elif self.framework == "tf":
                input_length = tf.shape(inputs["input_ids"])[-1].numpy()

            if input_length < self.model.config.min_length // 2:
                logger.warning(
                    "Your min_length is set to {}, but you input_length is only {}. You might consider decreasing min_length manually, e.g. summarizer('...', min_length=10)".format(
                        self.model.config.min_length, input_length
                    )
                )

            if input_length < self.model.config.max_length:
                logger.warning(
                    "Your max_length is set to {}, but you input_length is only {}. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=50)".format(
                        self.model.config.max_length, input_length
                    )
                )

            summaries = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generate_kwargs,
            )

            results = []
            for summary in summaries:
                record = {}
                if return_tensors:
                    record["summary_token_ids"] = summary
                if return_text:
                    record["summary_text"] = self.tokenizer.decode(
                        summary,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                results.append(record)
            return results


class TranslationPipeline(Pipeline):
    """
    Translates from one language to another.
    Usage::
        en_fr_translator = pipeline("translation_en_to_fr")
        en_fr_translator("How old are you?")
    Supported Models: "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"
    Arguments:
        model (:obj:`str` or :obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`, `optional`, defaults to :obj:`None`):
            The model that will be used by the pipeline to make predictions. This can be :obj:`None`, a string
            checkpoint identifier or an actual pre-trained model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
            If :obj:`None`, the default of the pipeline will be loaded.
        tokenizer (:obj:`str` or :obj:`~transformers.PreTrainedTokenizer`, `optional`, defaults to :obj:`None`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be :obj:`None`,
            a string checkpoint identifier or an actual pre-trained tokenizer inheriting from
            :class:`~transformers.PreTrainedTokenizer`.
            If :obj:`None`, the default of the pipeline will be loaded.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.
            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    def __call__(
        self,
        *texts,
        return_tensors=False,
        return_text=True,
        clean_up_tokenization_spaces=False,
        **generate_kwargs
    ):
        r"""
        Args:
            *texts: (list of strings) texts to be translated
            return_text: (bool, default=True) whether to add a decoded "translation_text" to each result
            return_tensors: (bool, default=False) whether to return the raw "translation_token_ids" to each result
            **generate_kwargs: extra kwargs passed to `self.model.generate`_
        Returns:
            list of dicts with 'translation_text' and/or 'translation_token_ids' for each text_to_translate
        .. _`self.model.generate`:
            https://huggingface.co/transformers/model_doc/bart.html#transformers.BartForConditionalGeneration.generate
        """
        assert (
            return_tensors or return_text
        ), "You must specify return_tensors=True or return_text=True"

        prefix = (
            self.model.config.prefix if self.model.config.prefix is not None else ""
        )

        if isinstance(texts[0], list):
            assert (
                self.tokenizer.pad_token_id is not None
            ), "Please make sure that the tokenizer has a pad_token_id when using a batch input"
            texts = ([prefix + text for text in texts[0]],)
            pad_to_max_length = True

        elif isinstance(texts[0], str):
            texts = (prefix + texts[0],)
            pad_to_max_length = False
        else:
            raise ValueError(
                " `documents[0]`: {} have the wrong format. The should be either of type `str` or type `list`".format(
                    texts[0]
                )
            )

        with self.device_placement():
            inputs = self._parse_and_tokenize(
                *texts, pad_to_max_length=pad_to_max_length
            )

            if self.framework == "pt":
                inputs = self.ensure_tensor_on_device(**inputs)
                input_length = inputs["input_ids"].shape[-1]

            elif self.framework == "tf":
                input_length = tf.shape(inputs["input_ids"])[-1].numpy()

            if input_length > 0.9 * self.model.config.max_length:
                logger.warning(
                    "Your input_length: {} is bigger than 0.9 * max_length: {}. You might consider increasing your max_length manually, e.g. translator('...', max_length=400)".format(
                        input_length, self.model.config.max_length
                    )
                )

            translations = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generate_kwargs,
            )
            results = []
            for translation in translations:
                record = {}
                if return_tensors:
                    record["translation_token_ids"] = translation
                if return_text:
                    record["translation_text"] = self.tokenizer.decode(
                        translation,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                results.append(record)
            return results


# Register all the supported task here
SUPPORTED_TASKS = {
    "feature-extraction": {
        "impl": FeatureExtractionPipeline,
        "tf": TFAutoModel if is_tf_available() else None,
        "pt": AutoModel if is_torch_available() else None,
        "default": {
            "model": {"pt": "distilbert-base-cased", "tf": "distilbert-base-cased"},
            "config": None,
            "tokenizer": "distilbert-base-cased",
        },
    },
    "sentiment-analysis": {
        "impl": TextClassificationPipeline,
        "tf": TFAutoModelForSequenceClassification if is_tf_available() else None,
        "pt": AutoModelForSequenceClassification if is_torch_available() else None,
        "default": {
            "model": {
                "pt": "distilbert-base-uncased-finetuned-sst-2-english",
                "tf": "distilbert-base-uncased-finetuned-sst-2-english",
            },
            "config": "distilbert-base-uncased-finetuned-sst-2-english",
            "tokenizer": "distilbert-base-uncased",
        },
    },
    "ner": {
        "impl": NerPipeline,
        "tf": TFAutoModelForTokenClassification if is_tf_available() else None,
        "pt": AutoModelForTokenClassification if is_torch_available() else None,
        "default": {
            "model": {
                "pt": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "tf": "dbmdz/bert-large-cased-finetuned-conll03-english",
            },
            "config": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "tokenizer": "bert-large-cased",
        },
    },
    "question-answering": {
        "impl": QuestionAnsweringPipeline,
        "pt": {
            "model": XLNetForQuestionAnswering,
            "config": XLNetConfig,
            "tokenizer": XLNetTokenizer,
        },
        "default": {
            "model": {
                "pt": "distilbert-base-cased-distilled-squad",
                "tf": "distilbert-base-cased-distilled-squad",
            },
            "config": None,
            "tokenizer": ("distilbert-base-cased", {"use_fast": False}),
        },
    },
    "fill-mask": {
        "impl": FillMaskPipeline,
        "tf": TFAutoModelWithLMHead if is_tf_available() else None,
        "pt": AutoModelWithLMHead if is_torch_available() else None,
        "default": {
            "model": {"pt": "distilroberta-base", "tf": "distilroberta-base"},
            "config": None,
            "tokenizer": ("distilroberta-base", {"use_fast": False}),
        },
    },
    "summarization": {
        "impl": SummarizationPipeline,
        "tf": TFAutoModelWithLMHead if is_tf_available() else None,
        "pt": AutoModelWithLMHead if is_torch_available() else None,
        "default": {
            "model": {"pt": "bart-large-cnn", "tf": None},
            "config": None,
            "tokenizer": ("bart-large-cnn", {"use_fast": False}),
        },
    },
    "translation_en_to_fr": {
        "impl": TranslationPipeline,
        "tf": TFAutoModelWithLMHead if is_tf_available() else None,
        "pt": AutoModelWithLMHead if is_torch_available() else None,
        "default": {
            "model": {"pt": "t5-base", "tf": "t5-base"},
            "config": None,
            "tokenizer": ("t5-base", {"use_fast": False}),
        },
    },
    "translation_en_to_de": {
        "impl": TranslationPipeline,
        "tf": TFAutoModelWithLMHead if is_tf_available() else None,
        "pt": AutoModelWithLMHead if is_torch_available() else None,
        "default": {
            "model": {"pt": "t5-base", "tf": "t5-base"},
            "config": None,
            "tokenizer": ("t5-base", {"use_fast": False}),
        },
    },
    "translation_en_to_ro": {
        "impl": TranslationPipeline,
        "tf": TFAutoModelWithLMHead if is_tf_available() else None,
        "pt": AutoModelWithLMHead if is_torch_available() else None,
        "default": {
            "model": {"pt": "t5-base", "tf": "t5-base"},
            "config": None,
            "tokenizer": ("t5-base", {"use_fast": False}),
        },
    },
}


def pipeline(
    task: str,
    model: Optional = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    framework: Optional[str] = None,
    **kwargs
) -> Pipeline:
    """
    Utility factory method to build a pipeline.
    Pipeline are made of:
        - A Tokenizer instance in charge of mapping raw textual input to token
        - A Model instance
        - Some (optional) post processing for enhancing model's output
    Args:
        task (:obj:`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:
            - "feature-extraction": will return a :class:`~transformers.FeatureExtractionPipeline`
            - "sentiment-analysis": will return a :class:`~transformers.TextClassificationPipeline`
            - "ner": will return a :class:`~transformers.NerPipeline`
            - "question-answering": will return a :class:`~transformers.QuestionAnsweringPipeline`
            - "fill-mask": will return a :class:`~transformers.FillMaskPipeline`
        model (:obj:`str` or :obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`, `optional`, defaults to :obj:`None`):
            The model that will be used by the pipeline to make predictions. This can be :obj:`None`, a string
            checkpoint identifier or an actual pre-trained model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
            If :obj:`None`, the default of the pipeline will be loaded.
        config (:obj:`str` or :obj:`~transformers.PretrainedConfig`, `optional`, defaults to :obj:`None`):
            The configuration that will be used by the pipeline to instantiate the model. This can be :obj:`None`,
            a string checkpoint identifier or an actual pre-trained model configuration inheriting from
            :class:`~transformers.PretrainedConfig`.
            If :obj:`None`, the default of the pipeline will be loaded.
        tokenizer (:obj:`str` or :obj:`~transformers.PreTrainedTokenizer`, `optional`, defaults to :obj:`None`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be :obj:`None`,
            a string checkpoint identifier or an actual pre-trained tokenizer inheriting from
            :class:`~transformers.PreTrainedTokenizer`.
            If :obj:`None`, the default of the pipeline will be loaded.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.
            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
    Returns:
        :class:`~transformers.Pipeline`: Class inheriting from :class:`~transformers.Pipeline`, according to
        the task.
    Examples::
        from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
        # Sentiment analysis pipeline
        pipeline('sentiment-analysis')
        # Question answering pipeline, specifying the checkpoint identifier
        pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')
        # Named entity recognition pipeline, passing in a specific model and tokenizer
        model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        pipeline('ner', model=model, tokenizer=tokenizer)
        # Named entity recognition pipeline, passing a model and configuration with a HTTPS URL.
        model_url = "https://s3.amazonaws.com/models.huggingface.co/bert/dbmdz/bert-large-cased-finetuned-conll03-english/pytorch_model.bin"
        config_url = "https://s3.amazonaws.com/models.huggingface.co/bert/dbmdz/bert-large-cased-finetuned-conll03-english/config.json"
        pipeline('ner', model=model_url, config=config_url, tokenizer='bert-base-cased')
    """
    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError(
            "Unknown task {}, available tasks are {}".format(
                task, list(SUPPORTED_TASKS.keys())
            )
        )

    framework = framework or get_framework(model)

    targeted_task = SUPPORTED_TASKS[task]
    # Try to infer modelcard from model or config name (if provided as str)
    if isinstance(model, str):
        modelcard = model

    # Instantiate model if needed
    framework = "pt"
    model_class = targeted_task[framework]["model"]
    config_class = targeted_task[framework]["config"]
    tokenizer_class = targeted_task[framework]["tokenizer"]
    task_class = targeted_task["impl"]
    if isinstance(model, str):
        modelcard = model
    model_name_or_path = model
    config = config_class.from_pretrained(model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
    model = model_class.from_pretrained(model_name_or_path, config=config)

    return task_class(
        model=model,
        tokenizer=tokenizer,
        modelcard=modelcard,
        framework=framework,
        task=task,
        **kwargs,
    )
