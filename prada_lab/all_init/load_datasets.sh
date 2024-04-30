# clone the LLM-Adapters repository (we forked the version we used)
git clone --depth=1 https://github.com/aryamanarora/LLM-Adapters.git /root/autodl-tmp/LLM-Adapters
# clone our own repository for holding ultrafeedback datasets
# git clone --depth=1 https://github.com/frankaging/ultrafeedback-datasets.git /root/autodl-tmp/ultrafeedback-datasets

# move datasetss
mv /root/autodl-tmp/LLM-Adapters/dataset/ /root/autodl-tmp/datasets/
mkdir /root/autodl-tmp/datasets/commonsense_170k
mv /root/autodl-tmp/LLM-Adapters/ft-training_set/commonsense_170k.json /root/autodl-tmp/datasets/commonsense_170k/train.json
mkdir /root/autodl-tmp/datasets/math_10k
mv /root/autodl-tmp/LLM-Adapters/ft-training_set/math_10k.json /root/autodl-tmp/datasets/math_10k/train.json
# mkdir /root/autodl-tmp/datasets/ultrafeedback
# mv /root/autodl-tmp/ultrafeedback-datasets/train.json /root/autodl-tmp/datasets/ultrafeedback/train.json

# clean
rm -rf /root/autodl-tmp/LLM-Adapters
# rm -rf /root/autodl-tmp/ultrafeedback-datasets
