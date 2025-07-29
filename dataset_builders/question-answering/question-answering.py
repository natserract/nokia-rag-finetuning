import datasets
import logging
import json

logger = logging.getLogger(__name__)

_DESCRIPTION = """\
A synthetic question answering dataset based on the Nokia 8 User Guide. It contains QA pairs
generated for instructional content such as device setup, SIM usage, fingerprint unlocking, and more.
"""
_URL = "dataset_builders/question-answering/data.json"

class DocQAConfig(datasets.BuilderConfig):
    """dataset config"""

    target_size: int = 1000
    max_size: int = 1000

    def __init__(self, **kwargs):
        super(DocQAConfig, self).__init__(**kwargs)

class DocQA(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
          DocQAConfig(
              name="docqa",
              version=datasets.Version("1.0.0", ""),
              description="Plain text",
          ),
      ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "id": datasets.Value("string"),
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }),
            supervised_keys=None,
            homepage="https://example.com",
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, #type: ignore
                gen_kwargs={
                    "filepath": _URL,
                },
            ),
        ]

    def _generate_examples(self, filepath): #type: ignore
        """This function returns the examples in the raw (text) form."""
        logger.info("Generating examples from = {}".format(filepath))

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for i, item in enumerate(data):
                yield i, {
                    "id": item["id"],
                    "question": item["question"],
                    "answer": item["answer"],
                }
