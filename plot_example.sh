################### Example for plotting Bias ##################
python spinupUtils/plot_bias.py \
    data/DDPG_value_estimation-HalfCheetah-v2-estimation/ \
    data/TD3_value_estimation-HalfCheetah-v2-estimation \
    --env HalfCheetah-v2 \
    -l  DDPG TD3 -s 0

################### Example for plotting reward ##################
python spinupUtils/plot_reward.py \
    data/DDPG_value_estimation-HalfCheetah-v2-reward/ \
    data/TD3_value_estimation-HalfCheetah-v2-reward \
    --env HalfCheetah-v2 \
    -l  DDPG TD3 -s 0