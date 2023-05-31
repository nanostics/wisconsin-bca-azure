# Pipeline

Code for the main Azure ML Pipeline (train-evaluate-register).

This is in the `lib` folder because I need them to share code amongst themselves (esp. from `constants.py`). To do this with files spread across `lib` as well as `azure/pipeline` would mean that we would have to upload the entire repository when running our ML pipelines.
