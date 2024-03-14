## Default Start
This will run the default task `arc_easy` with default model `mistralai/Mistral-7B-Instruct-v0.2`
```
docker run --gpus all  tybalex/lmeval
```


## Test a Local model
mount the local model dir and set the `MODEL_DIR` to the dir you mounted to. Example:
I have a local mistral rubra model at : `/home/paperspace/LLaMA-Factory/merged_model/mistral-rubra`
and I mount it to `/app/data/mistral-rubra`

```
docker run --gpus all -v /home/paperspace/LLaMA-Factory/merged_model/mistral-rubra:/app/data/mistral-rubra -e MODEL_DIR=/app/data/mistral-rubra  tybalex/lmeval
```
