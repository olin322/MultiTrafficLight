from functions import tenTrafficLightsRelativeDistanceSettings

def main(accumulated_epoc: int, epoc_this_run: int):
	env_id = "tenTrafficLightsRelativeDistanceSettings-v1"
	load_model_name = None
	# load_model_name = f"./models/PPO_tenTrafficLightsRelativeDistanceSettings_2048_3e-5_deltat_0.1_{accumulated_epoc}e8[-2,2]"
	trained_model_name = f"./models/PPO_tenTrafficLightsRelativeDistanceSettings_2048_3e-5_deltat_0.1_{accumulated_epoc + epoc_this_run}e8[-2,2]"
	training_iterations = epoc_this_run * 1e8
	tenTrafficLightsRelativeDistanceSettings(env_id, load_model_name, trained_model_name, training_iterations)

main(accumulated_epoc=0, epoc_this_run=20)

# python3 -m tensorboard.main --logdir=./tb_log/1201 port=6007