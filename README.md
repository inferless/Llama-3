# Tutorial - Deploy Llama-3 8B using Inferless
Meta releases the [Llama 3](https://llama.meta.com/llama3/), the latest open LLM models in the Llama family. The [Llama 3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6) models were trained on  8x more data on over 15 trillion tokens. It has a context length of 8K tokens and increases the vocabulary size of the tokenizer to tokenizer to 128,256 (from 32K tokens in the previous version).

In this tutorial we will deploy [LLama-3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B).

## TL;DR:

- Deployment of Meta-Llama-3-8B-hf model using [vLLM](https://github.com/vllm-project/vllm).
- You can expect an average latency of `1.63 sec` and throughput of 78.65 tokens per second. This setup has an average cold start time of `13.30 sec`.
- Dependencies defined in `inferless-runtime-config.yaml`.
- GitHub/GitLab template creation with `app.py`, `inferless-runtime-config.yaml` and `inferless.yaml`.
- Model class in `app.py` with `initialize`, `infer`, and `finalize` functions.
- Custom runtime creation with necessary system and Python packages.
- Model import via GitHub with `input_schema.py` file.
- Recommended GPU: NVIDIA A100 for optimal performance.
- Custom runtime selection in advanced configuration.
- Final review and deployment on the Inferless platform.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the Create new Runtime button. A pop-up will appear.

Next, provide a suitable name for your custom runtime and proceed by uploading the inferless-runtime.yaml file given above. Finally, ensure you save your changes by clicking on the save button.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the Add Model button.

Select the PyTorch as framework and choose **Repo(custom code)** as your model source and select your provider, and use the forked repo URL as the **Model URL**.

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/github-custom-code) for more information on model import.

---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. It has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

**Infer** - This function is where the inference happens. The argument to this function `inputs`, is a dictionary containing all the input parameters. The keys are the same as the name given in inputs. Refer to [input](#input) for more.

```python
def infer(self, inputs):
    prompt = inputs["prompt"]
```

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu by setting `self.pipe = None`.

For more information refer to the [Inferless docs](https://docs.inferless.com/).
