# Training Gym Style Guide

## Authoring a Tutorial

Each tutorial should use a [_Literate Programming_](https://en.wikipedia.org/wiki/Literate_programming) style and live in a single file. When tutorials are longer or may span multiple files, usually this speaks to one of the following:
1. The abstractions in the existing codebase are not enough: factor out common code into the `common` folder, framework specific common code in each framework's folder `frameworks/slime`, etc.
2. The tutorial being too complex: why were you asked to make this tutorial? Is there a way to achieve your goal while reducing complexity?

When updating common code, think about what other tutorials or configurations this could break, and make sure to rerun everything that is in the change path to validate.

### Validation should be cheap
When possible, limit the total number of steps. Tutorials should be cheap and easy to run.