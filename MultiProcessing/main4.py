from functions import tenTrafficLightsRelativeDistanceSettings

def main(accumulated_epoc: int, epoc_this_run: int):
	env_id = "tenTrafficLightsRelativeDistanceSettings-v2"
	# load_model_name = None
	load_model_name = f"./models/PPO_tenTrafficLightsRelativeDistanceSettingsTwoV2_2048_3e-5_deltat_0.1_{accumulated_epoc}e8[-2,2]"
	trained_model_name = f"./models/PPO_tenTrafficLightsRelativeDistanceSettingsTwoV2_2048_3e-5_deltat_0.1_{accumulated_epoc + epoc_this_run}e8[-2,2]"
	training_iterations = epoc_this_run * 1e8
	tenTrafficLightsRelativeDistanceSettings(env_id, load_model_name, trained_model_name, training_iterations)

main(accumulated_epoc=60, epoc_this_run=40)

# ./tb_log/1201 PPO_3  PPO_tenTrafficLightsRelativeDistanceSettingsTwoV2_2048_3e-5_deltat_0.1_20e8[-2,2]
# ./tb_log/1201 PPO_6  PPO_tenTrafficLightsRelativeDistanceSettingsTwoV2_2048_3e-5_deltat_0.1_40e8[-2,2]
# ./tb_log/1201 PPO_12 PPO_tenTrafficLightsRelativeDistanceSettingsTwoV2_2048_3e-5_delatt_0.1_60e8[-2,2]
# ./tb_log/1201 PPO_15 PPO_tenTrafficLightsRelativeDistanceSettingsTwoV2_2048_3e-5_delatt_0.1_100e8[-2,2]

# python3 -m tensorboard.main --logdir=./tb_log/1201 --port=6007