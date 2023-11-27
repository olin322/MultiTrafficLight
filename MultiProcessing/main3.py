from functions import tenTrafficLightsRelativeDistanceSettings

def main(accumulated_epoc: int, epoc_this_run: int):
	env_id = "tenTrafficLightsRelativeDistanceSettings-v1"
	# load_model_name = None
	load_model_name = f"./models/PPO_tenTrafficLightsRelativeDistanceSettings_2048_3e-5_deltat_0.1_{accumulated_epoc}e8[-2,2]"
	trained_model_name = f"./models/PPO_tenTrafficLightsRelativeDistanceSettings_2048_3e-5_deltat_0.1_{accumulated_epoc + epoc_this_run}e8[-2,2]"
	training_iterations = epoc_this_run * 1e8
	tenTrafficLightsRelativeDistanceSettings(env_id, load_model_name, trained_model_name, training_iterations)

main(accumulated_epoc=40, epoc_this_run=30)

# ./tb_log/1201 PPO_1  ./models/PPO_tenTrafficLightsRelativeDistanceSettings_2048_3e-5_deltat_0.1_20e8[-2,2]
# ./tb_log/1201 PPO_5  ./models/PPO_tenTrafficLightsRelativeDistanceSettings_2048_3e-5_deltat_0.1_40e8[-2,2]
# ./tb_log/1201 PPO_9  ./models/PPO_tenTrafficLightsRelativeDistanceSettings_2048_3e-5_deltat_0.1_80e8[-2,2] UNFINISHED DUE TO POWER OUTAGE
# ./tb_log/1201 PPO_11 ./models/PPO_tenTrafficLightsRelativeDistanceSettings_2048_3e-5_delatt_0.1_70e8[-2,2]
# python3 -m tensorboard.main --logdir=./tb_log/1201 --port=6007