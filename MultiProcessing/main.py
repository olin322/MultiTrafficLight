from functions import seventeenTrafficLights

def main(accumulated_epoc: int, epoc_this_run: int):
	load_model_name = f"./models/PPO_SeventeenTrafficLights_2048_3e-5_deltat_0.1_{accumulated_epoc}e8[-2,2]"
	trained_model_name = f"./models/PPO_SeventeenTrafficLights_2048_3e-5_deltat_0.1_{accumulated_epoc + epoc_this_run}e8[-2,2]"
	training_iterations = epoc_this_run * 1e8
	seventeenTrafficLights(load_model_name, trained_model_name, training_iterations)

main(accumulated_epoc=320, epoc_this_run=20)