from functions import seventeenTrafficLightsSettings

def main(accumulated_epoc: int, epoc_this_run: int):
	env_id = "SeventeenTrafficLights-v2"
	load_model_name = f"./models/PPO_SeventeenTrafficLightsV2_2048_3e-5_deltat_0.1_{accumulated_epoc}e8[-2,2]"
	trained_model_name = f"./models/PPO_SeventeenTrafficLightsV2_2048_3e-5_deltat_0.1_{accumulated_epoc + epoc_this_run}e8[-2,2]"
	training_iterations = epoc_this_run * 1e8
	seventeenTrafficLightsSettings(env_id, load_model_name, trained_model_name, training_iterations)

main(accumulated_epoc=320, epoc_this_run=40)

# PPO10 f"./models/PPO_SeventeenTrafficLightsV2_2048_3e-5_deltat_0.1_320e8[-2,2]
# PPO 2 f"./models/PPO_SeventeenTrafficLightsV2_2048_3e-5_deltat_0.1_360e8[-2,2]