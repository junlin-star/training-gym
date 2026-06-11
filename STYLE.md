# Training Gym Style Guide

## Authoring a Tutorial

Each tutorial should use a [_Literate Programming_](https://en.wikipedia.org/wiki/Literate_programming) style and live in a single file. When tutorials are longer or may span multiple files, usually this speaks to one of the following:
1. The abstractions in the existing codebase are not enough: factor out common code into the `common` folder, framework specific common code in each framework's folder `frameworks/slime`, etc.
2. The tutorial being too complex: why were you asked to make this tutorial? Is there a way to achieve your goal while reducing complexity?

When updating common code, think about what other tutorials or configurations this could break, and make sure to rerun everything that is in the change path to validate.

### Validation should be cheap
When possible, limit the total number of steps. Tutorials should be cheap and easy to run.

## Adding a new model

When adding a new model to the training-gym, you should first try adding the model as a `SlimeRecipe`. You can find examples in [slime model scripts](https://github.com/THUDM/slime/tree/main/scripts/models) or [slime examples](https://github.com/THUDM/slime/tree/main/examples).


`SlimeRecipe` by default use mbridge (`megatron_to_hf_mode=""`) instead of bridge (`megatron_to_hf_mode="bridge"`), which requires it to preconvert the weights. To determine if we should use bridge mode or mbridge, look upstream at the slime codebase at what was used for similar models.

### Model Naming Convention

Naming convention: For the model configuration, it should be `_` separated by model family identifiers and replacing `.` for versioning (e.g. `Qwen3_4B`, `Qwen3_6_35b`, `Kimi_K2_6`).
